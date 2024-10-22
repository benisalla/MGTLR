import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
from music_generator.model.RNN.from_scratch import RNNBlock
from music_generator.model.RNN.from_scratch.RNN import ACT_TYPES
from tqdm import tqdm
from torch.nn import functional as F

class MGRNN(nn.Module):
    def __init__(self, v_size: int, n_emb: int, h_dim: int, n_seq: int, act: ACT_TYPES = ACT_TYPES.TANH, bias: bool = False, drop_rate: float=0.0, device: str = "cpu"):
        super(MGRNN, self).__init__()
        self.h_dim: int = h_dim
        self.v_size: int = v_size
        self.n_emb: int = n_emb
        self.device: str = device
        self.bias: bool = bias
        self.n_seq: int = n_seq
        self.act: ACT_TYPES = act
        self.drop_rate: float = drop_rate

        self.emb: nn.Embedding = nn.Embedding(v_size, n_emb)   # [b, s] ==> [b, s, e]
        self.rnn: RNNBlock = RNNBlock(
            in_dim=n_emb,
            out_dim=h_dim,
            n_unit=n_seq,
            act=act,
            bias=bias,
            drop_rate=drop_rate,
            device=device
        )
        self.ln: nn.Linear = nn.Linear(h_dim, v_size)       # [b, s, e] ==> [b, s, v]

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x: torch.Tensor = self.emb(x).to(self.device)       # [b, s, e]
        x: torch.Tensor = self.rnn(x)                       # [b, s, h]

        if y is not None:
            logits: torch.Tensor = self.ln(x)               # [b, s, v]
            flog = logits.view(-1, self.v_size)             # [b * s, v]
            fy = y.view(-1)                                 # [b * s]
            loss: torch.Tensor = F.cross_entropy(flog, fy)
        else:
            logits: torch.Tensor = self.ln(x[:, [-1], :])   # [b, 1, v]
            loss: Optional[torch.Tensor] = None

        return logits, loss
    
    
    def generate(self, tokenizer, start: str, max_new_tokens: int = 1000, temperature: float = 1.0, top_k: Optional[int] = None) -> str:

        self.eval()
        input_eval = torch.tensor(tokenizer.encode(start), device=self.device).unsqueeze(0)
        generated: List[str] = []

        h: Optional[torch.Tensor] = None

        for _ in tqdm(range(max_new_tokens)):
            logits, h = self.forward(input_eval, h)
            preds = logits[:, -1, :] / temperature

            if top_k is not None:
                values, indices = torch.topk(preds, top_k)
                preds = torch.zeros_like(preds).scatter_(1, indices, values)

            probs = F.softmax(preds, dim=-1)
            id = torch.multinomial(probs, num_samples=1).item()

            input_eval = torch.tensor([[id]], device=self.device)
            generated.append(tokenizer.decode([id]))

        return start + ''.join(generated)

    def get_init_args(self) -> Dict[str, Any]:
        return {
            'v_size': self.v_size,
            'n_emb': self.n_emb,
            'h_dim': self.h_dim,
            'n_seq': self.n_seq,
            'act': self.act,
            'bias': self.bias,
            'drop_rate': self.drop_rate,
            'device': self.device
        }