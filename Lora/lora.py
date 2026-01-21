import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r):
        super(LoRALinear, self).__init__()
        self.in_features = in_features  # 对应 d
        self.out_features = out_features  # 对应 k
        self.r = r  # 低秩值

        # 原始权重矩阵，冻结
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False  # 冻结

        # LoRA 部分的参数，初始化 A 从均值为 0 的正态分布中采样，B 为全零
        self.A = nn.Parameter(torch.empty(r, in_features))  # 形状为 (r, d)
        self.B = nn.Parameter(torch.zeros(out_features, r))  # 形状为 (k, r)
        nn.init.normal_(self.A, mean=0.0, std=0.02)  # 初始化 A

        # 偏置项，可选
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # 原始部分
        original_output = torch.nn.functional.linear(x, self.weight, self.bias)
        # LoRA 增量部分
        delta_W = torch.matmul(self.B, self.A)  # 形状为 (k, d)
        lora_output = torch.nn.functional.linear(x, delta_W)
        # 总输出
        return original_output + lora_output