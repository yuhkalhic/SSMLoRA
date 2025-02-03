import torch
from torch import nn
import torch.nn.functional as F

class TimeAxis:
    def __init__(self, device):
        self.device = device
        self.time_axis = {}

    def get_current_time(self, key, length, features_num):
        if key not in self.time_axis:
            h_0 = torch.zeros((12, length, features_num), device=self.device)
            self.time_axis[key] = h_0
        return self.time_axis[key]

    def update_time(self, key, h_t):
        self.time_axis[key] = h_t.detach()

class SSMLoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, time_axis, layer_key, device):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.W_a = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.W_b = nn.Parameter(torch.zeros(rank, out_dim))
        self.W_c = nn.Parameter(torch.zeros(rank, rank))
        self.W_d = nn.Parameter(torch.zeros(rank, rank))
        self.alpha = alpha
        self.time_axis = time_axis
        self.layer_key = layer_key
        self.device = device
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        new_x = x @ self.W_a
        h_t = self.time_axis.get_current_time(self.layer_key, new_x.size()[1], new_x.size()[2])

        if h_t.size()[0] != new_x.size()[0]:
            if h_t.size()[0] < new_x.size()[0]:
                diff = new_x.size()[0] - h_t.size()[0]
                half_padding = torch.full((diff, h_t.size()[1], h_t.size()[2]), 0.5, device=self.device)
                h_t = torch.cat([h_t, half_padding], dim=0)
            else:
                indices = torch.randperm(h_t.size()[0])[:new_x.size()[0]]
                h_t = h_t[indices]

        if h_t.size()[1] > x.size()[1]:
            keep_indices = torch.randperm(h_t.size()[1], device=self.device)[:x.size()[1]]
            keep_indices, _ = torch.sort(keep_indices)
            h_t = h_t[:, keep_indices, :]
        elif h_t.size()[1] < x.size()[1]:
            padding = torch.zeros(h_t.size()[0], x.size()[1] - h_t.size()[1], h_t.size()[2], device=self.device)
            h_t = torch.cat([h_t, padding], dim=1)

        h_t1 = (h_t @ self.W_c + new_x @ self.W_d) + h_t
        min_val = h_t1.min(dim=2, keepdim=True)[0]
        max_val = h_t1.max(dim=2, keepdim=True)[0]
        h_t1_normed = (h_t1 - min_val) / (max_val - min_val + 1e-8)
        self.time_axis.update_time(self.layer_key, h_t1_normed.detach())
        y = self.alpha * ((new_x + F.leaky_relu(self.dropout(h_t1_normed))) @ self.W_b)
        return y

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha, time_axis, layer_key, device):
        super().__init__()
        self.linear = linear
        self.lora = SSMLoRALayer(
            linear.in_features, linear.out_features, rank, alpha, time_axis, layer_key, device
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)