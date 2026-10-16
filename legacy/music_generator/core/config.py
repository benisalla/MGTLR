import torch

class MGConfig:
    def __init__(self):
        self.lr_rate = 2e-4
        self.w_decay = 1e-2
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.amsgrad = False

        self.total_iters = 3000000
        self.n_step = 4
        self.num_epochs = 10000
        self.warmup_epochs = 300
        self.min_lr = 1e-6
        self.label_smoothing = 0.1

        self.n_embd = 128
        self.h_dim = 32
        self.n_block = 5
        self.v_size = None  
        self.b_size = 32
        self.ff_dim = 4 * self.n_embd
        self.drop_rate = 0.0
        self.bias = False
        self.max_seq_len = 1024
        self.seq_len = 512

        self.save_ckpt_path = ""
        self.load_ckpt_path = ""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_size = 0  
        
    def set_train_size(self, train_size):
        self.train_size = train_size
        self.warmup_iters = self.warmup_epochs * (self.train_size // self.b_size)
    
    def __repr__(self):
        return (f"<MusicGeneratorConfig save_ckpt_path={self.save_ckpt_path}, load_ckpt_path={self.load_ckpt_path}, "
                f"lr_rate={self.lr_rate}, w_decay={self.w_decay}, beta1={self.beta1}, beta2={self.beta2}, "
                f"eps={self.eps}, amsgrad={self.amsgrad}, total_iters={self.total_iters}, "
                f"num_epochs={self.num_epochs}, warmup_epochs={self.warmup_epochs}, device={self.device}, "
                f"warmup_iters={self.warmup_iters}, min_lr={self.min_lr}, label_smoothing={self.label_smoothing}, "
                f"n_embd={self.n_embd}, h_dim={self.h_dim}, n_block={self.n_block}, v_size={self.v_size}, "
                f"b_size={self.b_size}, ff_dim={self.ff_dim}, drop_rate={self.drop_rate}, bias={self.bias}, "
                f"max_seq_len={self.max_seq_len}, seq_len={self.seq_len}, train_size={self.train_size}>")



class LSTMConfig:
    def __init__(self, tokenizer, train_tokens):
        self.v_size = len(tokenizer.vocab)
        self.n_emb = 128
        self.b_size = 32
        self.h_dim = 256
        self.drop_rate = 0.0
        self.bias = False
        self.max_seq_len = 200
        self.seq_len = 200

        self.save_ckpt_path = ""
        self.load_ckpt_path = ""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.lr_rate = 1e-3
        self.w_decay = 1e-3
        self.beta1 = 0.9
        self.beta2 = 0.994
        self.eps = 1e-8
        self.amsgrad = False

        self.total_iters = 3000000
        self.n_step = 50
        self.num_epochs = 100000
        self.warmup_epochs = 300
        self.min_lr = 1e-6
        self.label_smoothing = 0.1

        self.set_train_size(len(train_tokens))

    def set_train_size(self, train_size):
        self.train_size = train_size
        self.warmup_iters = self.warmup_epochs * (self.train_size // self.b_size)

    def __repr__(self):
        return (f"<LSTMConfig save_ckpt_path={self.save_ckpt_path}, load_ckpt_path={self.load_ckpt_path}, "
                f"lr_rate={self.lr_rate}, w_decay={self.w_decay}, beta1={self.beta1}, beta2={self.beta2}, "
                f"eps={self.eps}, amsgrad={self.amsgrad}, total_iters={self.total_iters}, "
                f"num_epochs={self.num_epochs}, warmup_epochs={self.warmup_epochs}, device={self.device}, "
                f"warmup_iters={self.warmup_iters}, min_lr={self.min_lr}, label_smoothing={self.label_smoothing}, "
                f"n_emb={self.n_emb}, h_dim={self.h_dim}, v_size={self.v_size}, "
                f"b_size={self.b_size}, drop_rate={self.drop_rate}, bias={self.bias}, "
                f"max_seq_len={self.max_seq_len}, seq_len={self.seq_len}, train_size={self.train_size}>")
