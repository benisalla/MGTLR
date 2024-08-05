import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Optional, Dict, Tuple
from music_generator.model.LSTM.from_scratch import LSTMBlock
from tqdm import tqdm

class MGLSTM(nn.Module):
    def __init__(self, v_size: int, n_emb: int, h_dim: int, n_seq: int, bias: bool = False, device: str = "cpu"):
        super(MGLSTM, self).__init__()
        self.h_dim: int = h_dim
        self.v_size: int = v_size
        self.n_emb: int = n_emb
        self.device: str = device
        self.bias: bool = bias
        self.n_seq: int = n_seq

        self.emb: nn.Embedding = nn.Embedding(v_size, n_emb)
        self.lstm: LSTMBlock = LSTMBlock(
            in_dim=n_emb,
            out_dim=h_dim,
            n_unit=n_seq,
            bias=bias,
            device=device
        )
        self.fc: nn.Linear = nn.Linear(h_dim, v_size)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x: torch.Tensor = self.emb(x).to(self.device)  # [b, s, n_emb]

        x: torch.Tensor = self.lstm(x)  # [b, s, h_dim]

        if y is not None:
            logits: torch.Tensor = self.fc(x)  # [b, s, v_size]
            loss: torch.Tensor = F.cross_entropy(logits.view(-1, self.v_size), y.view(-1), ignore_index=-1)
        else:
            logits: torch.Tensor = self.fc(x[:, [-1], :])  # [b, 1, v_size]
            loss: Optional[torch.Tensor] = None

        return logits, loss

    def generate(self, tokenizer, start: str, length: int = 1000) -> str:
        self.eval()
        input_eval = torch.tensor(tokenizer.encode(start), device=self.device).unsqueeze(0)
        generated = []

        h = None
        for _ in tqdm(range(length)):
            logits, _ = self.forward(input_eval)
            preds = logits[:, -1, :]
            id = torch.multinomial(F.softmax(preds, dim=-1), num_samples=1).item()
            input_eval = torch.tensor([[id]], device=self.device)
            generated.append(tokenizer.decode([id]))

        return start + ''.join(generated)

    def get_init_args(self) -> Dict[str, Any]:
        return {
            'v_size': self.v_size,
            'n_emb': self.n_emb,
            'h_dim': self.h_dim,
            'n_seq': self.n_seq,
            'bias': self.bias
        }