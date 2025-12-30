import torch 
import torch.nn as nn 
# 计算scal 和 zero point 
def calcScaleZeroPoint(min_val, max_val, num_bits=8): 
    q_min = 0 
    q_max = 2**num_bits - 1 
    # 计算scale
    scale = (max_val - min_val) / (q_max - q_min) 
    # 计算zero point
    zero_point = q_max - max_val / scale  
    if zero_point < q_min:
        zero_point = q_min
    elif zero_point > q_max:
        zero_point = q_max 
    zero_point = int(zero_point)

    return scale, zero_point

def quantize_tensor(x, scale, zero_point, num_bits=8, signed=False):
    if signed:
        q_min = -2**(num_bits - 1)
        q_max = 2**(num_bits - 1) - 1
    else:
        q_min = 0
        q_max = 2**num_bits - 1
    # 量化
    x_q = x / scale + zero_point
    # Clip to quantized range
    # x_q = np.clip(x_q, q_min, q_max) 
    x_q = x_q.clip(q_min, q_max)
    # Round to nearest integer
    # x_q = np.round(x_q) 
    x_q = x_q.round()
    return x_q.float() 
def dequantize_tensor(q_x, scale, zero_point): 
    return scale *( q_x - zero_point) 

class QParam: 
    def __init__(self, int_number):
        self.int_number = int_number
        self.scale = None
        self.zero_point = None
        self.min = None
        self.max = None
    def update(self,tensor): 
        if self.min is None or self.min > tensor.min():
            self.min = tensor.min()
        if self.max is None or self.max < tensor.max():
            self.max = tensor.max()
        self.scale, self.zero_point = calcScaleZeroPoint(self.min, self.max) 
    def quantize_tensor(self, tensor):
        return quantize_tensor(tensor, self.scale, self.zero_point, self.int_number) 
    def dequantize_tensor(self, x_q):
        return dequantize_tensor(x_q, self.scale, self.zero_point) 
# 定义一个量化基类，这样可以减少一些重复代码，也能让代码结构更加清晰 
class QModule(nn.Module): 
    '''
    首先是 __init__ 函数，除了指定量化的位数外，还需指定是否提供量化输入 (qi) 及输出参数 (qo)。在前面也提到，不是每一个网络模块都需要统计输入的 min、max，大部分中间层都是用上一层的 qo 来作为自己的 qi 的，另外有些中间层的激活函数也是直接用上一层的 qi 来作为自己的 qi 和 qo。

    其次是 freeze 函数，这个函数会在统计完 min、max 后发挥作用。正如上文所说的，公式 (4) 中有很多项是可以提前计算好的，freeze 就是把这些项提前固定下来，同时也将网络的权重由浮点实数转化为定点整数。

    最后是 quantize_inference，这个函数主要是量化 inference 的时候会使用。实际 inference 的时候和正常的 forward 会有一些差异，可以根据之后的代码体会一下。'''    
    def __init__(self, qi=True, qo=True, num_bits=8):
        super(QModule, self).__init__()
        if qi:
            self.qi = QParam(num_bits)
        else:
            self.qi = None
        if qo:
            self.qo = QParam(num_bits)
        else:
            self.qo = None
    def freeze(self) : 
        pass 
    def quantize_inference(self, x):
        raise NotImplementedError('quantize_inference should be implemented.')


class QConv2d(QModule):
    '''
    这里的 M
    本来也需要通过移位来实现定点化加速，但 pytorch 中 bit shift 操作不好实现，因此我们还是用原始的乘法操作来代替。

    注意到 freeze 函数可能会传入 qi 或者 qo​，这也是之前提到的，有些中间的模块不会有自己的 qi，而是复用之前层的 qo 作为自己的 qi。

    接着是 forward 函数，这个函数和正常的 forward 一样，也是在 float 上进行的，只不过需要统计输入输出以及 weight 的 min、max 而已。有读者可能会疑惑为什么需要对 weight 量化到 int8 然后又反量化回 float，这里其实就是所谓的伪量化节点，因为我们在实际量化 inference 的时候会把 weight 量化到 int8，这个过程本身是有精度损失的 (来自四舍五入的 round 带来的截断误差)，所以在统计 min、max 的时候，需要把这个过程带来的误差也模拟进去。

    最后是 quantize_inference 函数，这个函数在实际 inference 的时候会被调用，对应的就是上面的公式 (7)。注意，这个函数里面的卷积操作是在 int 上进行的，这是量化推理加速的关键「当然，由于 pytorch 的限制，我们仍然是在 float 上计算，只不过数值都是整数。这也可以看出量化推理是跟底层实现紧密结合的技术」。

    理解 QConv2d 后，其他模块基本上异曲同工，这里不再赘述。
    '''
    def __init__(self, conv_module, qi=True, qo=True, num_bits=8):
        super(QConv2d, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.conv_module = conv_module
        self.qw = QParam(num_bits=num_bits)

    def freeze(self, qi=None, qo=None):
        
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M = self.qw.scale * self.qi.scale / self.qo.scale  # M = Sw*Sx/So

        self.conv_module.weight.data = self.qw.quantize_tensor(self.conv_module.weight.data)
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point

        self.conv_module.bias.data = quantize_tensor(self.conv_module.bias.data, scale=self.qi.scale * self.qw.scale, zero_point=0, signed=True) # Sb用Sx*Sw来替代

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)

        self.qw.update(self.conv_module.weight.data)

        self.conv_module.weight.data = self.qw.quantize_tensor(self.conv_module.weight.data)
        self.conv_module.weight.data = self.qw.dequantize_tensor(self.conv_module.weight.data)

        x = self.conv_module(x) 

        if hasattr(self, 'qo'):
            self.qo.update(x)

        return x
    
    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.fc_module(x)
        x = self.M * x + self.qo.zero_point
        return x


# 定义一个简单网络
class Net(nn.Module):

    def __init__(self, num_channels=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 40, 3, 1)
        self.conv2 = nn.Conv2d(40, 40, 3, 1, groups=20) # 这里用分组网络，可以增大量化带来的误差
        self.fc = nn.Linear(5*5*40, 10)
    def quantize(self, num_bits=8):
        self.qconv1 = QConv2d(self.conv1, qi=True, qo=True, num_bits=num_bits)
        self.qrelu1 = QReLU()
        self.qmaxpool2d_1 = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
        self.qconv2 = QConv2d(self.conv2, qi=False, qo=True, num_bits=num_bits)
        self.qrelu2 = QReLU()
        self.qmaxpool2d_2 = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits)
    def quantize_forward(self, x):
        x = self.qconv1(x)
        x = self.qrelu1(x)
        x = self.qmaxpool2d_1(x)
        x = self.qconv2(x)
        x = self.qrelu2(x)
        x = self.qmaxpool2d_2(x)
        x = x.view(-1, 5*5*40)
        x = self.qfc(x)
        return x

    def freeze(self):
        self.qconv1.freeze()
        self.qrelu1.freeze(self.qconv1.qo)
        self.qmaxpool2d_1.freeze(self.qconv1.qo)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qrelu2.freeze(self.qconv2.qo)
        self.qmaxpool2d_2.freeze(self.qconv2.qo)
        self.qfc.freeze(qi=self.qconv2.qo)
    def quantize_inference(self, x):
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qrelu1.quantize_inference(qx)
        qx = self.qmaxpool2d_1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qrelu2.quantize_inference(qx)
        qx = self.qmaxpool2d_2.quantize_inference(qx)
        qx = qx.view(-1, 5*5*40)
        qx = self.qfc.quantize_inference(qx)
        out = self.qfc.qo.dequantize_tensor(qx)
        return out
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*40)
        x = self.fc(x)
        return x


# 训练 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = torch.utils.data.DataLoader(
datasets.MNIST('data', train=True, download=True, 
                transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
datasets.MNIST('data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])),
batch_size=test_batch_size, shuffle=True, num_workers=1, pin_memory=True
)

model = Net().to(device)

# 后训练量化 
model = Net()
model.load_state_dict(torch.load('ckpt/mnist_cnn.pt'))
model.quantize(num_bits=8)
def direct_quantize(model, test_loader):
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.quantize_forward(data)
        if i % 200 == 0:
            break
    print('direct quantization finish')

model.freeze()

def quantize_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.quantize_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Quant Model Accuracy: {:.0f}%\n'.format(100. * correct / len(test_loader.dataset)))

quantize_inference(model, test_loader)

