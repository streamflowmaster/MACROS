import torch
import torch.nn as nn

def create_toeplitz_matrix_broadcast(n, W=1):
    # 生成索引差
    i = torch.arange(n).reshape(-1, 1)
    j = torch.arange(n).reshape(1, -1)
    matrix = ((i - j)**2) / W + 1 / W
    return matrix


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, similarity_matrix, alpha=0.5, reduction='mean'):
        super().__init__()
        self.S = torch.tensor(similarity_matrix, dtype=torch.float32)
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        # input: 模型输出的logits (shape: [batch_size, num_classes])
        # target: 真实类别的索引 (shape: [batch_size])

        # 计算标准交叉熵
        ce_loss = nn.functional.cross_entropy(input, target, reduction='none')

        # 获取预测类别（最大logit的类别）
        pred = torch.argmax(input, dim=1)

        # 为每个样本计算相似性权重
        batch_size = target.shape[0]
        weights = torch.ones(batch_size, dtype=torch.float32, device=input.device)

        for i in range(batch_size):
            y_true = target[i].item()
            y_pred = pred[i].item()
            weights[i] = 1 + self.alpha * self.S[y_true][y_pred]

        # 应用权重
        weighted_loss = ce_loss * weights

        # 根据reduction返回结果
        if self.reduction == 'mean':
            return torch.mean(weighted_loss)
        elif self.reduction == 'sum':
            return torch.sum(weighted_loss)
        else:
            return weighted_loss

class ToeplitzWeightedCrossEntropyLoss(nn.Module):
    """加权交叉熵损失，基于 Toeplitz 结构的平方距离加权，支持忽略特定标签。

    权重公式：w = 1 + alpha * ((y_true - y_pred)^2 / W + 1 / W)。
    适用于有序分类任务，忽略指定标签的损失计算。

    Args:
        num_classes (int): 类别数。
        W (float): 相似性公式的缩放因子，默认 1。
        alpha (float): 相似性权重的强度，默认 0.5。
        reduction (str): 损失聚合方式，'mean', 'sum' 或 'none'，默认 'mean'。
        ignore_index (int): 忽略的标签值，默认 -100。
        device (str or torch.device): 张量设备，默认 'cpu'。
    """
    def __init__(self, num_classes, W=1, alpha=0.5, reduction='mean', ignore_index=-100, device='cpu'):
        super().__init__()
        if W == 0:
            raise ValueError("W must be non-zero")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_classes = num_classes
        self.W = W
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.device = device

    def forward(self, input, target):
        """计算加权交叉熵损失。

        Args:
            input (torch.Tensor): 模型输出的 logits，形状 [batch_size, num_classes] 或 [batch_size, seq_len, num_classes]。
            target (torch.Tensor): 真实类别索引，形状 [batch_size] 或 [batch_size, seq_len]。

        Returns:
            torch.Tensor: 根据 reduction 参数聚合的损失。
        """
        # 支持多维输入（例如序列标注）
        if input.dim() > 2:
            input = input.view(-1, input.size(-1))  # [batch_size * seq_len, num_classes]
            target = target.view(-1)  # [batch_size * seq_len]

        # 验证 target 的标签
        valid_labels = (target >= 0) & (target < self.num_classes) | (target == self.ignore_index)
        if not valid_labels.all():
            raise ValueError("Target contains invalid labels")

        # 计算标准交叉熵，应用 ignore_index
        ce_loss = nn.functional.cross_entropy(input, target, reduction='none', ignore_index=self.ignore_index)

        # 获取预测类别
        pred = torch.argmax(input, dim=1)

        # 创建有效样本掩码
        valid_mask = target != self.ignore_index

        # 计算相似性值和权重
        S_values = ((target - pred)**2) / self.W + 1 / self.W
        weights = torch.ones_like(ce_loss, device=input.device)
        weights[valid_mask] = 1 + self.alpha * S_values[valid_mask]

        # 应用权重
        weighted_loss = ce_loss * weights

        # 根据 reduction 返回结果
        if self.reduction == 'mean':
            if valid_mask.sum() > 0:  # 避免除以零
                return weighted_loss[valid_mask].mean()
            else:
                return torch.tensor(0.0, device=input.device)
        elif self.reduction == 'sum':
            return weighted_loss[valid_mask].sum()
        else:
            return weighted_loss