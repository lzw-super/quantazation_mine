from torch.autograd import Function
import torch


class FakeQuantize(Function):
    """
    自定义的伪量化（Fake Quantization）操作，使用 Straight-Through Estimator (STE)。
    - 前向：对输入 x 进行量化再反量化（模拟量化噪声）
    - 反向：梯度直接传递（忽略量化操作的不可导性） 
    训练时不真正转换数据类型（仍用 float32），但模拟了量化引入的舍入误差，让网络学会适应量化噪声。
    """

    @staticmethod
    def forward(ctx, x, qparam):
        """
        前向传播：
        - ctx: 上下文对象，用于保存信息供 backward 使用（此处未用）
        - x: 输入张量（浮点）
        - qparam: 量化参数对象，需提供 quantize_tensor 和 dequantize_tensor 方法
        
        返回：量化后再反量化的 x（仍是浮点，但值已被“离散化”）
        """
        # 执行量化：将浮点 x 映射到整数量化格点（如 int8 范围）
        x_quantized = qparam.quantize_tensor(x)
        # 执行反量化：将整数格点值转回浮点表示（带量化误差）
        x_dequantized = qparam.dequantize_tensor(x_quantized)
        return x_dequantized

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播（Straight-Through Estimator）：
        - 梯度直接从输出传回输入，忽略量化操作
        - qparam 不需要梯度，返回 None
        
        返回：(grad_input, grad_qparam) → (grad_output, None)
        """
        return grad_output, None
def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    对输入 x 在给定的离散点 (xp, fp) 上进行分段线性插值。
    
    参数:
        x:  待插值的输入张量，任意 shape
        xp: 插值节点的 x 坐标（必须单调递增），shape [N]
        fp: 插值节点的 y 值，shape [N]，与 xp 对应
    
    返回:
        out: 与 x 同 shape 的插值结果张量
    """
    # 将 x 展平为 (B, L)，便于批量处理；保留原始 shape 用于最后恢复
    x_ = x.reshape(x.size(0), -1)  # shape: [batch_size, num_elements_per_sample]

    # 为 xp 和 fp 添加 batch 维度，便于广播
    xp = xp.unsqueeze(0)  # shape: [1, N]
    fp = fp.unsqueeze(0)  # shape: [1, N]

    # 计算每一段的斜率 m 和截距 b（线性方程 y = m*x + b）
    # xp[:,1:] - xp[:,:-1] 是相邻节点的 x 间距（Δx）
    # fp[:,1:] - fp[:,:-1] 是对应的 y 差值（Δy）
    m = (fp[:, 1:] - fp[:, :-1]) / (xp[:, 1:] - xp[:, :-1])  # shape: [1, N-1]
    b = fp[:, :-1] - (m * xp[:, :-1])                        # shape: [1, N-1]

    # 找出每个 x_ 元素落在哪个区间 [xp[i], xp[i+1])
    # torch.ge(x_, xp): 返回布尔矩阵，x_ >= xp 的位置为 True
    # .sum(-1) - 1: 统计每个 x 元素大于等于多少个 xp 节点 → 得到区间索引
    indices = torch.sum(torch.ge(x_[:, :, None], xp[:, None, :]), dim=-1) - 1  # shape: [B, L]

    # 防止越界：最小为 0，最大为 N-2（因为有 N-1 个区间）
    indices = torch.clamp(indices, 0, m.shape[-1] - 1)

    # 构造 batch 索引（行索引），用于高级索引 m[line_idx, indices]
    # 注意：原代码有 bug！下面给出修正版
    batch_size = x_.shape[0]
    line_idx = torch.arange(batch_size, device=indices.device).unsqueeze(1)  # shape: [B, 1]
    line_idx = line_idx.expand_as(indices)  # shape: [B, L]

    # 使用高级索引获取每个元素对应的斜率和截距
    slope = m[line_idx, indices]   # shape: [B, L]
    intercept = b[line_idx, indices]  # shape: [B, L]

    # 计算插值结果：y = m * x + b
    out = slope * x_ + intercept  # shape: [B, L]

    # 恢复原始 shape
    out = out.reshape(x.shape)
    return out