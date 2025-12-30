from torch.serialization import load
from model import *

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp


def direct_quantize(model, test_loader): 
    '''输入500个样本，进行伪量化前向传播，配合update函数确定整500个样本的输入输出以及中间权重的minmax、scale、zero_point'''
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.quantize_forward(data)
        if i % 500 == 0:
            break
    print('direct quantization finish')


def full_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Full Model Accuracy: {:.0f}%\n'.format(100. * correct / len(test_loader.dataset)))


def quantize_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.quantize_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Quant Model Accuracy: {:.0f}%\n'.format(100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    batch_size = 64
    using_bn = True
    load_quant_model_file = None
    # load_model_file = None

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=False, 
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
        batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

    if using_bn:
        model = NetBN()
        model.load_state_dict(torch.load('reference\pytorch-quantization-demo\ckpt\mnist_cnnbn.pt', map_location='cpu'))
        save_file = "reference\pytorch-quantization-demo\ckpt\mnist_cnnbn_ptq.pt"
    else:
        model = Net()
        model.load_state_dict(torch.load('reference\pytorch-quantization-demo\ckpt\mnist_cnn.pt', map_location='cpu'))
        save_file = "reference\pytorch-quantization-demo\ckpt\mnist_cnn_ptq.pt"

    model.eval()
    full_inference(model, test_loader)

    num_bits = 8
    model.quantize(num_bits=num_bits)
    model.eval()
    print('Quantization bit: %d' % num_bits)


    if load_quant_model_file is not None:
        model.load_state_dict(torch.load(load_quant_model_file))
        print("Successfully load quantized model %s" % load_quant_model_file)
    

    direct_quantize(model, train_loader)

    torch.save(model.state_dict(), save_file)
    model.freeze() # freeze函数是将模型的权重进行量化 然后

    # 测试是否设备转移是否正确
    # model.cuda()
    # print(model.qconv1.M.device)
    # model.cpu()
    # print(model.qconv1.M.device)

    quantize_inference(model, test_loader)

    



    
