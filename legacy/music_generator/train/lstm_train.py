import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from music_generator.core.config import LSTMConfig
from music_generator.data.dataloader import DataLoader
from music_generator.model.LSTM.from_scratch import MGLSTM
from music_generator.model.TRF.utils import live_plot_dual, save_checkpoints, train_epoch, validate_epoch
from music_generator.tokenizing.tokenizer.MGTokenizer import MGTokenizer

# Build configs
tokenizer = MGTokenizer()

# Data paths
main_dir = "./music_generator/src"
train_path = f'{main_dir}/dataset/train_abc.json'
val_path = f'{main_dir}/dataset/val_abc.json'
load_tokenizer_path = f"{main_dir}/tokenizer/mgt_tokenizer_v1.model"  

print("Loading the data...")
data_loader = DataLoader(train_path, val_path)

# Load and preprocess data
train_text, val_text = data_loader.get_data()
print(f"Train text sample: {train_text[:10]}")
print(f"Val text sample: {val_text[:10]}")

# Initialize and load tokenizer
tokenizer.load(load_tokenizer_path)

train_tokens = tokenizer.encode(train_text[:10000])
val_tokens = tokenizer.encode(val_text[:2000])

lstmconfig = LSTMConfig(tokenizer, train_tokens)
lstmconfig.set_train_size(len(train_text))

# Hyperparameters
lr_rate = lstmconfig.lr_rate
beta1, beta2 = lstmconfig.beta1, lstmconfig.beta2
eps = lstmconfig.eps
w_decay = lstmconfig.w_decay
num_epochs = lstmconfig.num_epochs
min_lr = lstmconfig.min_lr
device = lstmconfig.device
label_smoothing = lstmconfig.label_smoothing
n_embd = lstmconfig.n_emb
h_dim = lstmconfig.h_dim
v_size = lstmconfig.v_size
b_size = lstmconfig.b_size
max_seq_len = lstmconfig.max_seq_len
drop_rate = lstmconfig.drop_rate
bias = lstmconfig.bias
n_step = lstmconfig.n_step
n_seq = lstmconfig.seq_len
save_ckpt_path = lstmconfig.save_ckpt_path

# MGLSTM model
model = MGLSTM(
    v_size=v_size,
    n_emb=n_embd,
    h_dim=h_dim,
    n_seq=n_seq,
    bias=bias,
    device=device
)
model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

# Optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=lr_rate,
    betas=(beta1, beta2),
    eps=eps,
    weight_decay=w_decay
)

# Scheduler
scheduler = StepLR(optimizer, step_size=2000, gamma=0.000001)

print(f"len(train_tokens) : {len(train_tokens)}")
print(f"len(val_tokens) : {len(val_tokens)}")

# Start training
train_losses, train_accs, train_perplexities = [], [], []
val_losses, val_accs, val_perplexities = [], [], []

model.to(device)
curr_lr = optimizer.param_groups[0]['lr']

for epoch in range(num_epochs):
    t_loss, t_acc, t_perplexity = train_epoch(model, train_tokens, optimizer, n_seq, b_size, n_step, device)
    v_loss, v_acc, v_perplexity = validate_epoch(model, val_tokens, n_seq, b_size, n_step, device)


    scheduler.step()
    curr_lr = optimizer.param_groups[0]['lr']

    train_losses.append(t_loss)
    train_accs.append(t_acc)
    train_perplexities.append(t_perplexity)

    val_losses.append(v_loss)
    val_accs.append(v_acc)
    val_perplexities.append(v_perplexity)

    data_dict = {
        'Train Loss': train_losses,
        'Val Loss': val_losses,
        'Train Acc': train_accs,
        'Val Acc': val_accs,
        'Train Perplexity': train_perplexities,
        'Val Perplexity': val_perplexities
    }
    live_plot_dual(data_dict, title=f'Training & Validation Metrics [lr={curr_lr:.7f}]')

    torch.cuda.empty_cache()
    gc.collect()

    if epoch % 100 == 0 and epoch > 0:
        save_checkpoints(model, optimizer, save_ckpt_path)
