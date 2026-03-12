import json
import os
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_epoch(epoch, total_epochs, loss, acc, lr, elapsed):
    l = len(str(total_epochs))
    print(
        f"Epoch [{epoch+1:0{l}d}/{total_epochs}] - Loss: {loss:.4f}, Acc: {acc:.4f}, "
        f"LR: {lr:.6f}, Time: {elapsed:.2f}s"
    )


def save_results(clean_acc, corruption_mean_acc, severity_acc, save_path):
    # clean_acc: float
    # corruption_mean_acc: float
    # severity_acc: dict {1: float, 2: float, ..., 5: float}
    # save_path: str, JSON 파일 경로
    res = {
        "clean_acc": clean_acc,
        "corruption_mean_acc": corruption_mean_acc,
        "severity_acc": severity_acc
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(res, f, indent=4)
    return res