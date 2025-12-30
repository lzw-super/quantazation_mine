import copy
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

# ============================ Observer ============================
class ObserverBase(nn.Module):
    def __init__(self, q_level):
        super(ObserverBase, self).__init__()
        self.q_level = q_level

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        if self.q_level == "L":  # layer级(activation/weight)
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.q_level == "C":  # channel级(conv_weight)
            input = torch.flatten(input, start_dim=1)
            min_val = torch.min(input, 1)[0]
            max_val = torch.max(input, 1)[0]
        elif self.q_level == "FC":  # channel级(fc_weight)
            min_val = torch.min(input, 1, keepdim=True)[0]
            max_val = torch.max(input, 1, keepdim=True)[0]

        self.update_range(min_val, max_val)


class MinMaxObserver(ObserverBase):
    def __init__(self, q_level, out_channels):
        super(MinMaxObserver, self).__init__(q_level)
        self.num_flag = 0
        self.out_channels = out_channels
        if self.q_level == "L":
            self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
            self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))
        elif self.q_level == "C":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
        elif self.q_level == "FC":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )

    def update_range(self, min_val_cur, max_val_cur):
        if self.q_level == "C":
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = torch.min(min_val_cur, self.min_val)
            max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)


class MovingAverageMinMaxObserver(ObserverBase):
    def __init__(self, q_level, out_channels, momentum=0.1):
        super(MovingAverageMinMaxObserver, self).__init__(q_level)
        self.momentum = momentum
        self.num_flag = 0
        self.out_channels = out_channels
        if self.q_level == "L":
            self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
            self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))
        elif self.q_level == "C":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
        elif self.q_level == "FC":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )

    def update_range(self, min_val_cur, max_val_cur):
        if self.q_level == "C":
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)


class HistogramObserver(nn.Module):
    def __init__(self, q_level, momentum=0.1, percentile=0.9999):
        super(HistogramObserver, self).__init__()
        self.q_level = q_level
        self.momentum = momentum
        self.percentile = percentile
        self.num_flag = 0
        self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))

    @torch.no_grad()
    def forward(self, input):
        # MovingAveragePercentileCalibrator
        # PercentileCalibrator
        max_val_cur = torch.kthvalue(
            input.abs().view(-1), int(self.percentile * input.view(-1).size(0)), dim=0
        )[0]
        # MovingAverage
        if self.num_flag == 0:
            self.num_flag += 1
            max_val = max_val_cur
        else:
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
        self.max_val.copy_(max_val)
        

# ============================ Quantizer ============================
class Floor(Function):
    @staticmethod
    def forward(self, input, observer_min_val, observer_max_val, q_type):
        if q_type == 0:
            max_val = torch.max(torch.abs(observer_min_val), torch.abs(observer_max_val))
            min_val = -max_val
        else:
            max_val = observer_max_val
            min_val = observer_min_val
        self.save_for_backward(input, min_val, max_val)
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input))
        return output

    @staticmethod
    def backward(self, grad_output):
        input, min_val, max_val = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.gt(max_val)] = 0
        grad_input[input.lt(min_val)] = 0
        return grad_input, None, None, None
    
class Round(Function):
    @staticmethod
    def forward(self, input, observer_min_val, observer_max_val, q_type):
        if q_type == 0:
            max_val = torch.max(
                torch.abs(observer_min_val), torch.abs(observer_max_val)
            )
            min_val = -max_val
        else:
            max_val = observer_max_val
            min_val = observer_min_val
        self.save_for_backward(input, min_val, max_val)
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, min_val, max_val = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.gt(max_val)] = 0
        grad_input[input.lt(min_val)] = 0
        return grad_input, None, None, None



class Quantizer(nn.Module):
    def __init__(self, bits, observer, activation_weight_flag, qaft=False, union=False):
        super(Quantizer, self).__init__()
        self.bits = bits
        self.observer = observer
        self.activation_weight_flag = activation_weight_flag
        self.qaft = qaft
        self.union = union
        self.q_type = 0
        # scale/zero_point/eps
        if self.observer.q_level == "L":
            self.register_buffer("scale", torch.ones(1, dtype=torch.float32))
            self.register_buffer("zero_point", torch.zeros(1, dtype=torch.float32))
        elif self.observer.q_level == "C":
            self.register_buffer("scale", torch.ones((self.observer.out_channels, 1, 1, 1), dtype=torch.float32),)
            self.register_buffer("zero_point", torch.zeros((self.observer.out_channels, 1, 1, 1), dtype=torch.float32),)
        elif self.observer.q_level == "FC":
            self.register_buffer("scale", torch.ones((self.observer.out_channels, 1), dtype=torch.float32),)
            self.register_buffer("zero_point", torch.zeros((self.observer.out_channels, 1), dtype=torch.float32),)
        self.register_buffer("eps", torch.tensor(torch.finfo(torch.float32).eps, dtype=torch.float32))

    def update_qparams(self):
        raise NotImplementedError

    def round(self, input, observer_min_val, observer_max_val, q_type):
        output = Round.apply(input, observer_min_val, observer_max_val, q_type)
        return output

    def floor(self, input, observer_min_val, observer_max_val, q_type):
        output = Floor.apply(input, observer_min_val, observer_max_val, q_type)
        return output

    def forward(self, input):
        if self.bits > 32:
            output = input
        elif self.bits == 1:
            print(" Binary quantization is not supported ")
            assert self.bits != 1
        else:
            if not self.qaft:
                if self.training:  # QAT
                    if not self.union:
                        self.observer(input)        # update observer_min and observer_max
                    self.update_qparams()           # update scale and zero_point
                    
            if self.activation_weight_flag == 1:    # activation
                output = ((torch.clamp(self.round(  # floor or round?
                    input / self.scale.clone() + self.zero_point,
                    self.observer.min_val / self.scale + self.zero_point,
                    self.observer.max_val / self.scale + self.zero_point,
                    self.q_type, ), self.quant_min_val, self.quant_max_val, )
                    - self.zero_point) * self.scale.clone())
            elif self.activation_weight_flag == 0:  # weight or bias
                output = ((torch.clamp(self.round(
                    input / self.scale.clone() + self.zero_point,
                    self.observer.min_val / self.scale + self.zero_point,
                    self.observer.max_val / self.scale + self.zero_point,
                    self.q_type,),self.quant_min_val,self.quant_max_val,)
                    - self.zero_point) * self.scale.clone())
        return output


class SignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super(SignedQuantizer, self).__init__(*args, **kwargs)
        if self.activation_weight_flag == 0:  # weight
            self.register_buffer("quant_min_val", torch.tensor((-((1 << (self.bits - 1)) - 1)), dtype=torch.float32),)
            self.register_buffer("quant_max_val", torch.tensor(((1 << (self.bits - 1)) - 1), dtype=torch.float32),)
        elif self.activation_weight_flag == 1:  # activation
            self.register_buffer("quant_min_val", torch.tensor((-(1 << (self.bits - 1))), dtype=torch.float32),)
            self.register_buffer("quant_max_val", torch.tensor(((1 << (self.bits - 1)) - 1), dtype=torch.float32),)
        else:
            print("activation_weight_flag error")
            

class SymmetricQuantizer(SignedQuantizer):
    def update_qparams(self):
        self.q_type = 0
        quant_range = (float(self.quant_max_val - self.quant_min_val) / 2)  # quantized_range
        float_range = torch.max(torch.abs(self.observer.min_val), torch.abs(self.observer.max_val))  # float_range
        scale = float_range / quant_range  # scale
        scale = torch.max(scale, self.eps)  # processing for very small scale

        zero_point = torch.zeros_like(scale)  # zero_point
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)
        

class PowQuantizer(SignedQuantizer):
    def update_qparams(self):
        self.q_type = 0
        quant_range = (float(self.quant_max_val - self.quant_min_val) / 2)  # quantized_range
        float_range = torch.max(torch.abs(self.observer.min_val), torch.abs(self.observer.max_val))  # float_range
        scale = float_range / quant_range  # scale
        scale = torch.max(scale, self.eps)  # processing for very small scale
        scale = torch.pow(2, torch.floor(torch.log2(scale) + 0.5))  # pow scale
        zero_point = torch.zeros_like(scale)  # zero_point
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)
        

class UnsignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super(UnsignedQuantizer, self).__init__(*args, **kwargs)
        if self.activation_weight_flag == 0:  # weight
            self.register_buffer(
                "quant_min_val", torch.tensor((0), dtype=torch.float32)
            )
            self.register_buffer(
                "quant_max_val",
                torch.tensor(((1 << self.bits) - 2), dtype=torch.float32),
            )
        elif self.activation_weight_flag == 1:  # activation
            self.register_buffer(
                "quant_min_val", torch.tensor((0), dtype=torch.float32)
            )
            self.register_buffer(
                "quant_max_val",
                torch.tensor(((1 << self.bits) - 1), dtype=torch.float32),
            )
        else:
            print("activation_weight_flag error")


class AsymmetricQuantizer(UnsignedQuantizer):
    def update_qparams(self):
        self.q_type = 1
        quant_range = float(self.quant_max_val - self.quant_min_val)  # quantized_range
        float_range = self.observer.max_val - self.observer.min_val  # float_range
        scale = float_range / quant_range  # scale
        scale = torch.max(scale, self.eps)  # processing for very small scale
        sign = torch.sign(self.observer.min_val)
        zero_point = sign * torch.floor(
            torch.abs(self.observer.min_val / scale) + 0.5
        )  # zero_point
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)
        
def bias_quantize(input, scale):
    input = torch.floor(input / scale + 0.5)
    output = input * scale
    return output

# def bias_quantize_MCU(input, scale):
#     input = torch.floor(input / scale + 0.5)
#     shift_scale = torch.ceil(torch.log2(scale)).int()
#     adjusted_scale = torch.pow(2, shift_scale.float())
    
#     input = torch.round(input / scale)
#     output = input * scale
#     return output

# ============================ Modules ============================
class QuantConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        a_bits=8,
        w_bits=8,
        qaft=False,
    ):
        super(QuantConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.input_quantizer = PowQuantizer(
            bits=self.a_bits,
            observer=MinMaxObserver(q_level="L", out_channels=None),
            # observer=MovingAverageMinMaxObserver(q_level="L", out_channels=None),
            # observer=HistogramObserver(q_level="L"),
            activation_weight_flag=1,
            qaft=qaft,)

        self.weight_quantizer = PowQuantizer(
            bits=self.w_bits,
            observer=MinMaxObserver(q_level="L", out_channels=None),
            # observer=MovingAverageMinMaxObserver(q_level="L", out_channels=None),
            # observer=HistogramObserver(q_level="L"),
            activation_weight_flag=0,
            qaft=qaft,)

    def __repr__(self):
        return (f"{self.__class__.__name__}( "
                f"weight_quantizer={type(self.weight_quantizer).__name__}, w_bits={self.w_bits}, "
                f"input_quantizer={type(self.input_quantizer).__name__}, a_bits={self.a_bits} )")
        
    def forward(self, input):
        quant_input = self.input_quantizer(input)
        quant_weight = self.weight_quantizer(self.weight)
        quant_bias = bias_quantize(self.bias, (self.weight_quantizer.scale * self.input_quantizer.scale))      
        output = F.conv2d(
            input=quant_input,
            weight=quant_weight,
            # bias = self.bias,
            bias=quant_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        # print(quant_input)
        # print(output)
        # quant_output = self.activation_quantizer(output)
        return output
    
    
class QuantLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        a_bits=8,
        w_bits=8,
        qaft=False,
    ):
        super(QuantLinear, self).__init__(
            in_features, 
            out_features, 
            bias,
        )
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.input_quantizer = PowQuantizer(
            bits=self.a_bits,
            observer=MinMaxObserver(q_level="L", out_channels=None),
            # observer=MovingAverageMinMaxObserver(q_level="L", out_channels=None),
            # observer=HistogramObserver(q_level="L"),
            activation_weight_flag=1,
            qaft=qaft,)

        self.weight_quantizer = PowQuantizer(
            bits=self.w_bits,
            observer=MinMaxObserver(q_level="L", out_channels=None),
            # observer=MovingAverageMinMaxObserver(q_level="L", out_channels=None),
            # observer=HistogramObserver(q_level="L"),
            activation_weight_flag=0,
            qaft=qaft,)
        
        self.output_quantizer = PowQuantizer(
            bits=self.a_bits,
            observer=MinMaxObserver(q_level="L", out_channels=None),
            # observer=MovingAverageMinMaxObserver(q_level="L", out_channels=None),
            # observer=HistogramObserver(q_level="L"),
            activation_weight_flag=1,
            qaft=qaft,)

    def __repr__(self):
        return (f"{self.__class__.__name__}( "
                f"weight_quantizer={type(self.weight_quantizer).__name__}, w_bits={self.w_bits},"
                f"input_quantizer={type(self.input_quantizer).__name__}, a_bits={self.a_bits},"
                f"output_quantizer={type(self.output_quantizer).__name__}, a_bits={self.a_bits})")

    def forward(self, input):
        quant_input = self.input_quantizer(input)
        quant_weight = self.weight_quantizer(self.weight)
        quant_bias = bias_quantize(self.bias, (self.weight_quantizer.scale * self.input_quantizer.scale))
        output = F.linear(
            input=quant_input,
            weight=quant_weight,
            # bias = self.bias,
            bias=quant_bias
            )
        quant_output = self.output_quantizer(output)
        # exit()
        return quant_output

# class QuantMaxPool2d(nn.MaxPool2d):
#     def __init__(
#         self,
#         kernel_size,
#         stride=None,
#         # padding=0,
#         a_bits=8,
#         qaft=False,
#     ):
#         super(QuantMaxPool2d, self).__init__(
#             kernel_size,
#             stride,
#             # padding,
#         )
#         self.a_bits = a_bits
#         self.activation_quantizer = PowQuantizer(
#             bits=self.a_bits,
#             observer=MinMaxObserver(q_level="L", out_channels=None),
#             # observer=MovingAverageMinMaxObserver(q_level="L", out_channels=None),
#             # observer=HistogramObserver(q_level="L"),
#             activation_weight_flag=1,
#             qaft=qaft,)
        
#     def __repr__(self):
#         return (f"{self.__class__.__name__}( "
#                 f"activation_quantizer={type(self.activation_quantizer).__name__}, a_bits={self.a_bits} )")
        
#     def forward(self, input):
#         output = F.max_pool2d(
#             input=input,
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             # self.padding,
#         )
#         quant_output = self.activation_quantizer(output)
#         return quant_output    

class QuantAvgPool2d(nn.AvgPool2d):
    def __init__(
        self,
        kernel_size,
        stride=None,
        # padding=0,
        a_bits=8,
        qaft=False,
    ):
        super(QuantAvgPool2d, self).__init__(
            kernel_size,
            stride,
            # padding,
        )
        self.a_bits = a_bits
        self.activation_quantizer = PowQuantizer(
            bits=self.a_bits,
            observer=MinMaxObserver(q_level="L", out_channels=None),
            # observer=MovingAverageMinMaxObserver(q_level="L", out_channels=None),
            # observer=HistogramObserver(q_level="L"),
            activation_weight_flag=1,
            qaft=qaft,)
        
    def __repr__(self):
        return (f"{self.__class__.__name__}( "
                f"input_quantizer={type(self.activation_quantizer).__name__}, a_bits={self.a_bits} )")
    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.avg_pool2d(
            input=quant_input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            # self.padding,
        )
        # quant_output = self.activation_quantizer(output)
        return output
    

class QuantAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(
        self,
        output_size,
        a_bits=8,
        qaft = False,
    ):
        super(QuantAdaptiveAvgPool2d, self).__init__(output_size)
        self.a_bits = a_bits
        self.activation_quantizer = PowQuantizer(
            bits=self.a_bits,
            observer=MinMaxObserver(q_level="L", out_channels=None),
            # observer=MovingAverageMinMaxObserver(q_level="L", out_channels=None),
            # observer=HistogramObserver(q_level="L"),
            activation_weight_flag=1,
            qaft=qaft,)
        
    def __repr__(self):
        return (f"{self.__class__.__name__}( "
                f"input_quantizer={type(self.activation_quantizer).__name__}, a_bits={self.a_bits} )")
        
    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.adaptive_avg_pool2d(
            input=quant_input,
            output_size=self.output_size)
        # quant_output = self.activation_quantizer(output)
        return output
    

def add_quant_op(
    module,
    a_bits=1,
    w_bits=1,
    qaft=False,
):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            if child.bias is not None:
                quant_conv = QuantConv2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=True,
                    padding_mode=child.padding_mode,
                    a_bits=a_bits,
                    w_bits=w_bits,
                    qaft=qaft,
                )
                quant_conv.bias.data = child.bias
            else:
                quant_conv = QuantConv2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=False,
                    padding_mode=child.padding_mode,
                    a_bits=a_bits,
                    w_bits=w_bits,
                    qaft=qaft,
                )
            quant_conv.weight.data = child.weight
            module._modules[name] = quant_conv
        elif isinstance(child, nn.Linear):
            if child.bias is not None:
                quant_linear = QuantLinear(
                    child.in_features,
                    child.out_features,
                    bias=True,
                    a_bits=a_bits,
                    w_bits=w_bits,
                    qaft=qaft,
                )
                quant_linear.bias.data = child.bias
            else:
                quant_linear = QuantLinear(
                    child.in_features,
                    child.out_features,
                    bias=False,
                    a_bits=a_bits,
                    w_bits=w_bits,
                    qaft=qaft,
                )
            quant_linear.weight.data = child.weight
            module._modules[name] = quant_linear
        elif isinstance(child, nn.AvgPool2d):
            quant_avg_pool = QuantAvgPool2d(
                kernel_size=child.kernel_size,
                stride=child.stride,
                # padding=child.padding,
                a_bits=a_bits,
                qaft=qaft,
            )
            module._modules[name] = quant_avg_pool
        # elif isinstance(child, nn.MaxPool2d):
        #     quant_max_pool = QuantMaxPool2d(
        #         kernel_size=child.kernel_size,
        #         stride=child.stride,
        #         # padding=child.padding,
        #         a_bits=a_bits,
        #         qaft=qaft,
        #     )
        #     module._modules[name] = quant_max_pool
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            quant_adaptive_avg_pool = QuantAdaptiveAvgPool2d(
                output_size=child.output_size,
                a_bits=a_bits,
                qaft=qaft,
            )
            module._modules[name] = quant_adaptive_avg_pool
        else:
            add_quant_op(
                child,
                a_bits=a_bits,
                w_bits=w_bits,
                qaft=qaft,
            )
        
            
def prepare(
    model,
    inplace=False,
    a_bits=8,
    w_bits=8,
    qaft=False,
):
    if not inplace:
        model = copy.deepcopy(model)
    add_quant_op(
        module=model,
        a_bits=a_bits,
        w_bits=w_bits,
        qaft=qaft,
    )
    return model