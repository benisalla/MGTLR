import gc
import torch
import torch.nn as nn
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import math
from torch.nn.utils import clip_grad_norm_
from music_generator.data.utils import get_batch


def lr_schedule(epoch, learning_rate, warmup_iters, total_iters, min_lr):
    if epoch < warmup_iters:
        return min_lr + (learning_rate - min_lr) * epoch / warmup_iters
    elif epoch > total_iters:
        return min_lr
    else:
        decay_ratio = (epoch - warmup_iters) / (total_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)


def save_checkpoints(model, optimizer, epoch, save_ckpt_path):
    model_args = model.get_init_args()
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "epoch": epoch,
    }
    torch.save(checkpoint, save_ckpt_path)
    print("\033[94mCheckpoints Saved Successfully :)\033[0m")


def live_plot_dual(data_dict, figsize=(12, 5), title=""):
    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(data_dict["Train Loss"], "r-", label="Train Loss")
    ax1.plot(data_dict["Val Loss"], "b-", label="Val Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    if data_dict["Train Loss"]:
        ax1.annotate(
            f"{data_dict['Train Loss'][-1]:.4f}",
            xy=(1, data_dict["Train Loss"][-1]),
            xytext=(8, 0),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
        )
    if data_dict["Val Loss"]:
        ax1.annotate(
            f"{data_dict['Val Loss'][-1]:.4f}",
            xy=(1, data_dict["Val Loss"][-1]),
            xytext=(8, 0),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
        )

    ax2.plot(data_dict["Train Acc"], "r-", label="Train Accuracy")
    ax2.plot(data_dict["Val Acc"], "b-", label="Val Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    if data_dict["Train Acc"]:
        ax2.annotate(
            f"{data_dict['Train Acc'][-1]:.4f}",
            xy=(1, data_dict["Train Acc"][-1]),
            xytext=(8, 0),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
        )
    if data_dict["Val Acc"]:
        ax2.annotate(
            f"{data_dict['Val Acc'][-1]:.4f}",
            xy=(1, data_dict["Val Acc"][-1]),
            xytext=(8, 0),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
        )

    plt.suptitle(title)
    plt.show()


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_metrics(epoch, num_epochs, t_loss, t_acc, t_perplexity, v_loss, v_acc, v_perplexity):
    print(f"{Colors.HEADER}Epoch {epoch + 1}/{num_epochs}{Colors.ENDC}")
    print(
        f"{Colors.OKGREEN}Training:{Colors.ENDC}   Loss: {t_loss:.4f}, Accuracy: {t_acc:.4f}, Perplexity: {t_perplexity:.4f}"
    )
    print(
        f"{Colors.OKCYAN}Validation:{Colors.ENDC} Loss: {v_loss:.4f}, Accuracy: {v_acc:.4f}, Perplexity: {v_perplexity:.4f}"
    )
    print("-" * 60)


def memory_usage():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9} GB")


def accuracy(logits, targets):
    predicted = logits.argmax(-1)
    correct_predictions = (predicted == targets).float()
    accuracy = correct_predictions.mean()
    return accuracy.item()


def perplexity(loss):
    return torch.exp(loss).item()

def train_epoch(model, train_tokens, optimizer, seq_len, b_size, n_step, device):
    model.train()
    total_loss = 0
    total_perplexity = 0
    total_accuracy = 0
    steps = 0

    for _ in range(n_step):
        x, y = get_batch(train_tokens, seq_len=seq_len, b_size=b_size)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits, loss = model(x, y)
        loss.backward()

        clip_grad_norm_(model.parameters(), 0.05)
        optimizer.step()

        total_loss += loss.item()
        total_perplexity += perplexity(loss)
        total_accuracy += accuracy(logits, y)
        steps += 1

        del x, y, logits, loss
        gc.collect()

    avg_loss = total_loss / steps
    avg_perplexity = total_perplexity / steps
    avg_accuracy = total_accuracy / steps
    return avg_loss, avg_accuracy, avg_perplexity


def validate_epoch(model, val_tokens, seq_len, b_size, n_step, device):
    model.eval()
    total_loss = 0
    total_perplexity = 0
    total_accuracy = 0
    steps = 0

    with torch.no_grad():
        for _ in range(n_step):
            x, y = get_batch(val_tokens, seq_len=seq_len, b_size=b_size)
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)

            total_loss += loss.item()
            total_perplexity += perplexity(loss)
            total_accuracy += accuracy(logits, y)
            steps += 1

            del x, y, logits, loss

    avg_loss = total_loss / steps
    avg_perplexity = total_perplexity / steps
    avg_accuracy = total_accuracy / steps

    return avg_loss, avg_accuracy, avg_perplexity