# overview of quantization 
## linear quantization (TFlite) 
### post quantization 
[TF lite paper(Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference )](https://arxiv.org/abs/1712.05877) 
- 只需要统计原始浮点数的max、min以及规定的量化位数就可以计算出scal以及zero，然后进行量化。scal一般是浮点数（可用一维来转换成整数运算），zero一般是整数，用于量化实数0。
![linear quantization equal](mark_img\quantization_equal.png "linear quantization equal") 
- 此外一些小数可以用整数来移位得到，只要误差能够接受即可。
![network_exa](mark_img\network_exa.png "network example")
- 因此，在最简单的后训练量化算法中，我们会先按照正常的 forward 流程跑一些数据，在这个过程中，统计输入输出以及中间 feature map 的 min、max。等统计得差不多了，我们就可以根据 min、max 来计算 scale 和 zero point，然后根据公式 (4) 中的，对一些数据项提前计算。
- 之后，在 inference 的时候，我们会先把输入 x量化成定点整数qx，然后按照公式 (4) 计算卷积的输出qa1，这个结果依然是整型的，然后继续计算 relu 的输出 qa2。对于 fc 层来说，它本质上也是矩阵运算，因此也可以用公式 (4) 计算，然后得到 qy。最后，根据 fc 层已经计算出来的 scale 和zero point，推算回浮点实数 y。除了输入输出的量化和反量化操作，其他流程完全可以用定点运算来完成。
[pytorch quantization demo code](https://github.com/Jermmy/pytorch-quantization-demo) 
![quan flow](mark_img\quan_flow.png "quantization flow")
### Quantitative Perception Training 
量化感知训练，顾名思义，就是在量化的过程中，对网络进行训练，从而让网络参数能更好地适应量化带来的信息损失。这种方式更加灵活，因此准确性普遍比后训练量化要高。
- Straight Through Estimator(STE):直接跳过伪量化的过程，避开 round。直接把卷积层的梯度回传到伪量化之前的 weight 上。这样一来，由于卷积中用的 weight 是经过伪量化操作的，因此可以模拟量化误差，把这些误差的梯度回传到原来的 weight，又可以更新权重，使其适应量化产生的误差，量化训练就可以正常进行下去了。
- 之前的代码中输入输出没有加伪量化节点，这在后训练量化中没有问题，但在量化训练中最好加上，方便网络更好地感知量化带来的损失。
- 在 bit = 1 的时候，我发现量化训练回传的梯度为 0，训练基本失败了。这是因为 bit = 1 的时候，整个网络已经退化成一个二值网络了，而低比特量化训练本身不是一件容易的事情，虽然我们前面用 STE 解决了梯度的问题，但由于低比特会使得网络的信息损失巨大，因此通常的训练方式很难起到作用。
- 低比特量化的时候，学习率太高容易导致梯度变为 0，导致量化训练完全不起作用。
- 