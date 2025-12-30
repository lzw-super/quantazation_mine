import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from model.Quantize import prepare
from model.transform import *
# from model.model_bnfuse import *
import numpy as np

torch.set_printoptions(
    precision=5,
    threshold=float('inf'),
    linewidth=120,
    sci_mode=False
)

class DS37K_wav2_bn(nn.Module):
    def __init__(self, num_classes=2):
        super(DS37K_wav2_bn, self).__init__()
        self.block0 = self.depthwise_separable_block(16, 32, 3, 2, 1)
        self.block1 = self.depthwise_separable_block(32, 32, 3, 1, 1)
        self.block2 = self.depthwise_separable_block(32, 64, 3, 2, 1)
        self.block3 = self.depthwise_separable_block(64, 64, 3, 1, 1)
        self.block4 = self.depthwise_separable_block(64, 128, 3, 2, 1)
        self.block5 = self.depthwise_separable_block(128, 128, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(7,7)
        self.flatten = nn.Flatten()
        # self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def depthwise_separable_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True))

    def forward(self, x):
        y0 = wavlet_2_func(x) / 128
        y1 = self.block0(y0)
        y2 = self.block1(y1)        
        y3 = self.block2(y2)
        y4 = self.block3(y3)
        y5 = self.block4(y4)
        y6 = self.block5(y5)
        y7 = self.maxpool2(y6)
        y8 = self.flatten(y7)
        y9 = self.fc(y8)
        return y0,y1,y2,y3,y4,y5,y6,y7,y8,y9

def softmax(x):
    max_val, _ = torch.max(x, dim=1, keepdim=True)
    x_stable = x - max_val                # 与 C 中的 fa - max_val 对齐
    exp_x = torch.exp(x_stable)
    sum_exp = torch.sum(exp_x, dim=1, keepdim=True)
    output = (exp_x[:, 1:2] / sum_exp )* 100   # C 代码中返回的是 b 对应的 softmax 值
    return output

model_pth = './save_pth_qat/20250824/DS37K_wav2_bn_2253/best_model_epoch489.pth'
input_path = './mcu_simulate/Live611_th250/input_005/input_005.pth'
# input_path = './mcu_simulate/Fake.pth'
input = torch.load(input_path).to(torch.float32).to('cuda')
# print(input.shape)
# exit()

net = DS37K_wav2_bn(num_classes=2).to('cuda')
net = prepare(
    model=net,
    inplace=True,
    a_bits=8,
    w_bits=8,
    qaft=True,).to('cuda')
if model_pth is not None:
    net.load_state_dict(torch.load(model_pth))
        
cnn = torch.load(model_pth)
layer_config = [# (layer_name, has_weight, has_bias)
    ('block0.0',         True,  True ),         
    ('block0.1',         True,  True ),         
    ('block1.0',         True,  True ),   
    ('block1.1',         True,  True ),
    ('block2.0',         True,  True ),
    ('block2.1',         True,  True ),
    ('block3.0',         True,  True ),
    ('block3.1',         True,  True ),
    ('block4.0',         True,  True ),
    ('block4.1',         True,  True ),
    ('block5.0',         True,  True ),
    ('block5.1',         True,  True ),
    ('fc',               True,  True ),         
]
# Generate quantized weight and bias parameters
quantized_params = {}
for layer_name, has_weight, has_bias in layer_config:
    if has_weight:  
        weight_key = f"{layer_name}.weight"
        weight_float = cnn[weight_key]
        weight_scale = cnn[f"{layer_name}.weight_quantizer.scale"]
        weight_int8 = torch.floor(weight_float / weight_scale + 0.5).clamp(-128, 127)
        quantized_params[weight_key] = weight_int8

    if has_bias:  
        input_scale = cnn[f"{layer_name}.input_quantizer.scale"]
        bias_scale = weight_scale * input_scale
        bias_key = f"{layer_name}.bias"
        bias_float = cnn[bias_key]
        bias_int32 = torch.floor(bias_float / bias_scale + 0.5)
        quantized_params[f"{bias_key}_int32"] = bias_int32
        
        max_abs_bias = torch.max(torch.abs(bias_int32))
        shift_scale = torch.ceil(torch.log2(max_abs_bias / 128)).item() if max_abs_bias > 127 else 0
        bias_int8 = (bias_int32 / (2 ** shift_scale)).round().clamp(-128, 127)
        quantized_params[f"{bias_key}_int32"] = bias_int32
        quantized_params[f"{bias_key}_int8"] = bias_int8
        quantized_params[f"{bias_key}_lscale"] = int(shift_scale)
        # quantized_params[f"{bias_key}_MCU"] = bias_int8 * (2**(int(shift_scale)))
        quantized_params[f"{bias_key}_MCU"] = bias_int32


def quantize(input, scale, q_min, q_max):
    input = torch.floor((input / scale) +0.5)
    # input = input//scale
    output = torch.clamp(input, q_min, q_max)
    return output

y0,y1,y2,y3,y4,y5,y6,y7,y8,y9 = net(input)
# =========================================================================================================================================================
# q_input = torch.floor(input / cnn['block0.0.input_quantizer.scale'] + 0.5).clamp(-128, 127)
q_input = input
q_input = wavlet_2_hardware(q_input)
# print(q_input.shape,'\n',q_input.permute(0,2,3,1)[:,:,:10,:10])
# exit()
scale0_0 = cnn['block0.1.input_quantizer.scale'] / (cnn['block0.0.weight_quantizer.scale'] * cnn['block0.0.input_quantizer.scale'])
print('scale0_0',math.log2(scale0_0))
block0_0 = F.conv2d(q_input, weight=quantized_params['block0.0.weight'], bias=quantized_params['block0.0.bias_MCU'], groups=16, stride=2, padding=1)
q_block0_0 = quantize(block0_0, scale0_0, q_min=-128, q_max=127)
# print('q_block0_0.shape,'\n',q_block0_0.permute(0,2,3,1)[:,:10,:10,:])
# exit()
scale0_1 = cnn['block1.0.input_quantizer.scale'] / (cnn['block0.1.weight_quantizer.scale'] * cnn['block0.1.input_quantizer.scale'])
print('scale0_1',math.log2(scale0_1))
block0_1 = F.conv2d(q_block0_0, weight=quantized_params['block0.1.weight'], bias=quantized_params['block0.1.bias_MCU'], stride=1, padding=0)
q_block0_1 = quantize(block0_1, scale0_1, q_min=-128, q_max=127)
q_block0_2 = F.relu(q_block0_1)
# print('q_block0_2.shape','\n',q_block0_2.permute(0,2,3,1)[:,:1,:50,:])
# exit()
t1 = torch.floor(y1 / cnn['block1.0.input_quantizer.scale'] + 0.5)
# print(torch.sum(q_block0_2),torch.sum(t1))
# exit()
# =========================================================================================================================================================
scale1_0 = cnn['block1.1.input_quantizer.scale'] / (cnn['block1.0.weight_quantizer.scale'] * cnn['block1.0.input_quantizer.scale'])
print('scale1_0',math.log2(scale1_0))
block1_0 = F.conv2d(q_block0_2, weight=quantized_params['block1.0.weight'], bias=quantized_params['block1.0.bias_MCU'], groups=32, stride=1, padding=1)
q_block1_0 = quantize(block1_0, scale1_0, q_min=-128, q_max=127)
# print(q_block1_0.shape,'\n',q_block1_0.permute(0,2,3,1)[:,:10,:10,:])
# exit()
scale1_1 = cnn['block2.0.input_quantizer.scale'] / (cnn['block1.1.weight_quantizer.scale'] * cnn['block1.1.input_quantizer.scale'])
print('scale1_1',math.log2(scale1_1))
block1_1 = F.conv2d(q_block1_0, weight=quantized_params['block1.1.weight'], bias=quantized_params['block1.1.bias_MCU'], stride=1, padding=0)
q_block1_1 = quantize(block1_1, scale1_1, q_min=-128, q_max=127)
q_block1_2 = F.relu(q_block1_1)
# print(q_block1_2.shape,'\n',q_block1_2.permute(0,2,3,1)[:,:10,:10,:])
# exit()

# t2 = torch.floor(y2 / cnn['block2.0.input_quantizer.scale'] + 0.5)
# print(torch.sum(q_block1_2),torch.sum(t2))
# exit()
# =========================================================================================================================================================
scale2_0 = cnn['block2.1.input_quantizer.scale'] / (cnn['block2.0.weight_quantizer.scale'] * cnn['block2.0.input_quantizer.scale'])
print('scale2_0',math.log2(scale2_0))
block2_0 = F.conv2d(q_block1_2, weight=quantized_params['block2.0.weight'], bias=quantized_params['block2.0.bias_MCU'], groups=32, stride=2, padding=1)
# print(quantized_params['block2.0.weight'].permute(1,2,3,0))
# exit()
q_block2_0 = quantize(block2_0, scale2_0, q_min=-128, q_max=127)
# print(block2_0.shape,'\n',block2_0.permute(0,2,3,1)[:,:10,:10,:])
scale2_1 = cnn['block3.0.input_quantizer.scale'] / (cnn['block2.1.weight_quantizer.scale'] * cnn['block2.1.input_quantizer.scale'])
print('scale2_1',math.log2(scale2_1))
block2_1 = F.conv2d(q_block2_0, weight=quantized_params['block2.1.weight'], bias=quantized_params['block2.1.bias_MCU'], stride=1, padding=0)
q_block2_1 = quantize(block2_1, scale2_1, q_min=-128, q_max=127)
q_block2_2 = F.relu(q_block2_1)
# print(q_block2_2.shape,'\n',q_block2_2.permute(0,2,3,1)[:,:10,:10,:])
# exit()
# t3 = q_block2_2 * cnn['block3.0.input_quantizer.scale']
# print(torch.sum(y3),torch.sum(t3))
# exit()
# =========================================================================================================================================================
scale3_0 = cnn['block3.1.input_quantizer.scale'] / (cnn['block3.0.weight_quantizer.scale'] * cnn['block3.0.input_quantizer.scale'])
print('scale3_0',math.log2(scale3_0))
block3_0 = F.conv2d(q_block2_2, weight=quantized_params['block3.0.weight'], bias=quantized_params['block3.0.bias_MCU'], groups=64, stride=1, padding=1)
q_block3_0 = quantize(block3_0, scale3_0, q_min=-128, q_max=127)
# print(q_block3_0.shape,'\n',q_block3_0.permute(0,2,3,1)[:,:10,:10,:])
# exit()
scale3_1 = cnn['block4.0.input_quantizer.scale'] / (cnn['block3.1.weight_quantizer.scale'] * cnn['block3.1.input_quantizer.scale'])
print('scale3_1',math.log2(scale3_1))
block3_1 = F.conv2d(q_block3_0, weight=quantized_params['block3.1.weight'], bias=quantized_params['block3.1.bias_MCU'], stride=1, padding=0)

q_block3_1 = quantize(block3_1, scale3_1, q_min=-128, q_max=127)
q_block3_2 = F.relu(q_block3_1)
# print(q_block3_2.shape,'\n',q_block3_2.permute(0,2,3,1)[:,:10,:10,:])
# exit()
# t4 = q_block3_2 * cnn['block4.0.input_quantizer.scale']
# print(torch.sum(y4),torch.sum(t4))
# exit()
# =========================================================================================================================================================
scale4_0 = cnn['block4.1.input_quantizer.scale'] / (cnn['block4.0.weight_quantizer.scale'] * cnn['block4.0.input_quantizer.scale'])
print('scale4_0',math.log2(scale4_0))
block4_0 = F.conv2d(q_block3_2, weight=quantized_params['block4.0.weight'], bias=quantized_params['block4.0.bias_MCU'], groups=64, stride=2, padding=1)
q_block4_0 = quantize(block4_0, scale4_0, q_min=-128, q_max=127)
# print(q_block4_0.shape,'\n',q_block4_0.permute(0,2,3,1)[:,:10,:10,:])
# exit()
scale4_1 = cnn['block5.0.input_quantizer.scale'] / (cnn['block4.1.weight_quantizer.scale'] * cnn['block4.1.input_quantizer.scale'])
print('scale4_1',math.log2(scale4_1))
block4_1 = F.conv2d(q_block4_0, weight=quantized_params['block4.1.weight'], bias=quantized_params['block4.1.bias_MCU'], stride=1, padding=0)

q_block4_1 = quantize(block4_1, scale4_1, q_min=-128, q_max=127)
q_block4_2 = F.relu(q_block4_1)
# print(q_block4_2.shape,'\n',q_block4_2.permute(0,2,3,1)[:,:10,:10,:])
# exit()
# t5 = q_block4_2 * cnn['block5.0.input_quantizer.scale']
# print(torch.sum(y5),torch.sum(t5))
# exit()
# =========================================================================================================================================================
scale5_0 = cnn['block5.1.input_quantizer.scale'] / (cnn['block5.0.weight_quantizer.scale'] * cnn['block5.0.input_quantizer.scale'])
print('scale5_0',math.log2(scale5_0))
block5_0 = F.conv2d(q_block4_2, weight=quantized_params['block5.0.weight'], bias=quantized_params['block5.0.bias_MCU'], groups=128, stride=1, padding=1)
q_block5_0 = quantize(block5_0, scale5_0, q_min=-128, q_max=127)
# print(q_block3_0.shape,'\n',q_block3_0.permute(0,2,3,1)[:,:10,:10,:])
# exit()
scale5_1 = cnn['fc.input_quantizer.scale'] / (cnn['block5.1.weight_quantizer.scale'] * cnn['block5.1.input_quantizer.scale'])
print('scale5_1',math.log2(scale5_1))
block5_1 = F.conv2d(q_block5_0, weight=quantized_params['block5.1.weight'], bias=quantized_params['block5.1.bias_MCU'], stride=1, padding=0)

q_block5_1 = quantize(block5_1, scale5_1, q_min=-128, q_max=127)
q_block5_2 = F.relu(q_block5_1)
print(q_block5_2.shape,'\n',q_block5_2.permute(0,2,3,1)[:,:,:,:])
# exit()
# t6 = q_block5_2 * cnn['fc.output_quantizer.scale']
# print(torch.sum(y6),torch.sum(t6))
# exit()
# =========================================================================================================================================================
max = F.max_pool2d(q_block5_2, 7, 7)
# print(max.shape)
flatten = max.view(-1,128)
# print(flatten)
# exit()
# y_f = torch.floor(y9 / cnn['fc.input_quantizer.scale'] + 0.5)
# print(flatten,'\n',y_f)

scale_fc = cnn['fc.output_quantizer.scale'] / (cnn['fc.input_quantizer.scale'] * cnn['fc.weight_quantizer.scale'])
print('scale-fc',math.log2(scale_fc))
fc = F.linear(flatten, weight=quantized_params['fc.weight'], bias=quantized_params['fc.bias_MCU'])
q_fc = quantize(fc, scale_fc, -128, 127)
# y_q = torch.floor(y9 / cnn['fc.output_quantizer.scale'] + 0.5)
pred = softmax(q_fc)
print(q_fc,pred)

# if pred.dim() > 1:
    # pred = pred.squeeze(-1)
# count_0 = torch.sum(pred < 50).item()  
# count_1 = torch.sum(pred >= 50).item() 
# predictions = torch.argmax(q_fc, dim=1)

# print(f"预测为 0 的个数: {count_0}")
# print(f"预测为 1 的个数: {count_1}")