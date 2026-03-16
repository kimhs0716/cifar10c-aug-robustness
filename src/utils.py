import json
import os
import random
import numpy as np
import torch


def resolve_device(device_str):
    if device_str == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        if torch.xpu.is_available():
            print("WARNING: CUDA not available, falling back to XPU")
            return "xpu"
        print("WARNING: CUDA not available, falling back to cpu")
        return "cpu"
    if device_str == "xpu":
        if torch.xpu.is_available():
            return "xpu"
        print("WARNING: XPU not available, falling back to cpu")
        return "cpu"
    return "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_epoch(epoch, total_epochs, train_loss, train_acc, test_acc, lr, elapsed):
    l = len(str(total_epochs))
    print(
        f"Epoch [{epoch+1:0{l}d}/{total_epochs}] - Loss: {train_loss:.4f}, "
        f"Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
        f"LR: {lr:.6f}, Time: {elapsed:.2f}s"
    )


def save_results(clean_acc, corruption_res, save_dir):
    res = dict()
    res["clean_acc"] = clean_acc
    severity_acc = 0
    for per_sev in corruption_res.values():
        severity_acc += sum(per_sev.values()) / len(per_sev)
    res["corruption_mean_acc"] = severity_acc / len(corruption_res)
    res["severity_acc"] = {
        s: sum(corruption_res[c][s] for c in corruption_res) / len(corruption_res)
        for s in range(1, 6)
    }
    mce = sum(1 - acc for per_sev in corruption_res.values() for acc in per_sev.values()) / (len(corruption_res) * 5)
    res["mce"] = round(mce, 4)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=4)
    with open(os.path.join(save_dir, "corruption_metrics.csv"), "w") as f:
        f.write("corruption,severity,acc\n")
        for c in corruption_res:
            for s in corruption_res[c]:
                f.write(f"{c},{s},{corruption_res[c][s]:.4f}\n")

    print(f"Results saved to \"{save_dir}\" successfully.")

    return res
    