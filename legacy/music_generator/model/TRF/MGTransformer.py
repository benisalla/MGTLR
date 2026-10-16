import torch
from torch import nn
import torch.nn.functional as F
import math
from music_generator.model.TRF.MGBlock import MGBlock
from music_generator.model.TRF.MGLayerNorm import MGLayerNorm

class MGTransformer(nn.Module):
    def __init__(self, n_embd: int, h_dim: int, n_block: int, v_size: int, b_size: int, max_seq_len: int, ff_dim: int = None, drop_rate: float = 0.0, bias: bool = False, device: str = "cpu"):
        super(MGTransformer, self).__init__()
        self.n_block: int = n_block
        self.ff_dim: int = ff_dim if ff_dim else 4 * n_embd
        self.n_embd: int = n_embd
        self.h_dim: int = h_dim
        self.max_seq_len: int = max_seq_len
        self.n_head: int = n_embd // h_dim
        self.v_size: int = v_size
        self.b_size: int = b_size
        self.drop_rate: float = drop_rate
        self.bias: bool = bias
        self.device: str = device

        self.decoder: nn.ModuleDict = nn.ModuleDict(dict(
            tok_emb = nn.Embedding(self.v_size, self.n_embd),
            pos_emb = nn.Embedding(self.max_seq_len, self.n_embd),
            dropout = nn.Dropout(self.drop_rate),
            blocks = nn.ModuleList([
                MGBlock(n_embd=n_embd, n_head=self.n_head, b_size=b_size, ff_dim=self.ff_dim, drop_rate=drop_rate, bias=bias)
                for _ in range(n_block)
            ]),
            f_ln = MGLayerNorm(n_dim=self.n_embd, bias=self.bias),
        ))

        self.lm_head: nn.Linear = nn.Linear(self.n_embd, self.v_size, bias=False)
        self.decoder.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)

        for param_name, param in self.named_parameters():
            if param_name.endswith('proj.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * n_block))
        # print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.decoder.pos_emb.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
            self, 
            idx: torch.Tensor, 
            targets: torch.Tensor = None
        ) -> (torch.Tensor, torch.Tensor):
        
        device = idx.device
        B, T = idx.size()
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length T = {T} exceeds maximum allowed seq_len = {self.seq_len}")

        pos: torch.Tensor = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb: torch.Tensor = self.decoder.tok_emb(idx)
        pos_emb: torch.Tensor = self.decoder.pos_emb(pos)

        x = self.decoder.dropout(tok_emb + pos_emb)
        for block in self.decoder.blocks:
            x = block(x)
        x = self.decoder.f_ln(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def get_init_args(self):
        return {
            'n_embd': self.n_embd,
            'h_dim': self.h_dim,
            'n_block': self.n_block,
            'v_size': self.v_size,
            'b_size': self.b_size,
            'max_seq_len': self.max_seq_len,
            'drop_rate': self.drop_rate,
            'bias': self.bias
        }

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.b_size else idx[:, -self.b_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx