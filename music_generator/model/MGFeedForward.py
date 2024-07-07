import torch.nn as nn

class MGFeedForward(nn.Module):
    def __init__(self, n_embd, ff_dim, drop_rate, bias):
        super().__init__()
        self.fc_ln = nn.Linear(n_embd, ff_dim, bias=bias)
        self.act = nn.GELU()
        self.proj = nn.Linear(ff_dim, n_embd, bias=bias)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.act(self.fc_ln(x))
        x = self.dropout(self.proj(x))
        return x