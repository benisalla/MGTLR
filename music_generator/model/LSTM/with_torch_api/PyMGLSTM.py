import torch
from torch import nn
from typing import List, Optional, Tuple
import torch.nn.functional as F
from tqdm import tqdm
from music_generator.model.LSTM.with_torch_api import PyLSTM

class PyMGLSTM(nn.Module):
    def __init__(self, v_size: int, n_emb: int, h_dim: int, n_seq: int, bias: bool = False, device: str = "cpu") -> None:
        super(PyMGLSTM, self).__init__()
        self.bias: bool = bias
        self.h_dim: int = h_dim
        self.v_size: int = v_size
        self.n_emb: int = n_emb
        self.n_seq: int = n_seq
        self.device: str = device

        self.emb: nn.Embedding = nn.Embedding(v_size, n_emb)
        self.lstm: PyLSTM = PyLSTM(in_dim=n_emb, out_dim=h_dim, n_unit=n_seq, bias=bias)
        self.fc: nn.Linear = nn.Linear(h_dim, v_size)

    def forward(self, 
                x: torch.Tensor, y: Optional[torch.Tensor] = None, 
                h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if h is None:
            h = (torch.zeros(self.n_seq, x.size(0), self.h_dim, device=self.device),
                torch.zeros(self.n_seq, x.size(0), self.h_dim, device=self.device))

        x = self.emb(x).to(self.device)

        x, h = self.lstm(x, h)

        logits: torch.Tensor = self.fc(x)

        if y is not None:
            loss: torch.Tensor = F.cross_entropy(logits.view(-1, self.v_size), y.view(-1), ignore_index=-1)
        else:
            logits = logits[:, -1, :]
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