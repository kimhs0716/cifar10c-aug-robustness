import os
import time

import yaml
import torch
import torch.nn as nn

from data import get_cifar10_loaders, get_cifar10c_loader
from model import get_model
from utils import set_seed, log_epoch, save_results


CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
    "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise",
    "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise",
    "snow", "spatter", "speckle_noise", "zoom_blur",
]


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        outputs = model(xb)
        preds = outputs.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    
    return correct / total


def evaluate_corruption(model, data_dir, mean, std, batch_size, device):
    results = dict()

    for corruption in CORRUPTIONS:
        results[corruption] = dict()
        for severity in range(1, 6):
            loader = get_cifar10c_loader(
                data_dir, corruption, severity,
                mean, std, batch_size, device
            )
            acc = evaluate(model, loader, device)
            results[corruption][severity] = acc
    
    return results


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    SEED = cfg["experiment"]["seed"]
    device = cfg["experiment"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    mean = cfg["data"]["normalization"]["mean"]
    std = cfg["data"]["normalization"]["std"]
    data_dir = cfg["data"]["data_root"]
    epochs = cfg["train"]["epochs"]
    batch_size = cfg["data"]["batch_size"]
    c10c_dir = cfg["eval"]["robustness"]["root"]
    output_dir = cfg["output"]["dir"]
    run_name = cfg["output"]["run_name"]

    set_seed(SEED)

    train_loader, test_loader = get_cifar10_loaders(
        data_dir, mean, std, batch_size,
        aug_type=cfg["data"]["aug_type"], device=device
    )

    model = get_model(cfg["model"]["name"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), **cfg["train"]["optimizer"]
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, 
        end_factor=1.0, total_iters=5
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs-5,
        eta_min=cfg["train"]["scheduler"]["min_lr"]
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine],
        milestones=[5]
    )

    best_acc = 0.0

    for epoch in range(epochs):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer,
            criterion, device
        )

        test_acc = evaluate(model, test_loader, device)

        log_epoch(
            epoch, epochs, train_loss, train_acc, test_acc,
            optimizer.param_groups[0]["lr"], time.time() - start
        )

        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt_dir = os.path.join(output_dir, run_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'acc': best_acc
            }, f"{ckpt_dir}/best.pt")
    
    ckpt = torch.load(f"{ckpt_dir}/best.pt")
    model.load_state_dict(ckpt['model'])
    clean_acc = evaluate(model, test_loader, device)
    corruption_result = evaluate_corruption(
        model, c10c_dir, mean, std,
        batch_size, device
    )

    all_accs = [acc for per_sev in corruption_result.values() for acc in per_sev.values()]
    corruption_mean_acc = sum(all_accs) / len(all_accs)

    severity_acc = {
        s: sum(corruption_result[c][s] for c in corruption_result) / len(corruption_result)
        for s in range(1, 6)
    }

    save_path = f"{cfg['output']['dir']}/{cfg['output']['run_name']}/metrics.json"
    save_results(clean_acc, corruption_mean_acc, severity_acc, save_path)


if __name__ == "__main__":
    main("configs/baseline.yaml")
