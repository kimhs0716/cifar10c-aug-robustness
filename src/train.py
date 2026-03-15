import os
import time
import shutil

import yaml
import torch
import torch.nn as nn

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from data import get_cifar10_loaders, CIFAR10CSlice
from model import get_model
from utils import resolve_device, set_seed, log_epoch, save_results


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


def evaluate_corruption(model, data_dir, mean, std, batch_size, device, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    pin_memory = device == "cuda"
    results = dict()

    labels = np.load(os.path.join(data_dir, "labels.npy"))
    for corruption in CORRUPTIONS:
        all_data = np.load(os.path.join(data_dir, f"{corruption}.npy"))
        results[corruption] = dict()
        for severity in range(1, 6):
            start = (severity - 1) * 10000
            end = severity * 10000
            dataset = CIFAR10CSlice(all_data[start:end], labels[start:end], transform)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
            acc = evaluate(model, loader, device)
            results[corruption][severity] = acc

    return results


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    SEED = cfg["experiment"]["seed"]
    device = resolve_device(cfg["experiment"]["device"])
    mean = cfg["data"]["normalization"]["mean"]
    std = cfg["data"]["normalization"]["std"]
    data_dir = cfg["data"]["data_root"]
    epochs = cfg["train"]["epochs"]
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]
    c10c_dir = cfg["eval"]["robustness"]["root"]
    output_dir = cfg["output"]["dir"]
    run_name = cfg["output"]["run_name"]
    save_interval = cfg["output"]["save_interval"]
    save_dir = os.path.join(output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    shutil.copy(config_path, os.path.join(save_dir, "config.yaml"))

    set_seed(SEED)

    train_loader, test_loader = get_cifar10_loaders(
        data_dir, mean, std, batch_size,
        aug_type=cfg["data"]["aug_type"], device=device, num_workers=num_workers
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
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'acc': best_acc
            }, f"{save_dir}/best.pt")
        
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'acc': test_acc
            }, f"{save_dir}/last.pt")
    
    ckpt = torch.load(f"{save_dir}/best.pt")
    model.load_state_dict(ckpt['model'])
    clean_acc = evaluate(model, test_loader, device)
    corruption_result = evaluate_corruption(
        model, c10c_dir, mean, std,
        batch_size, device, num_workers
    )

    save_results(clean_acc, corruption_result, save_dir)


if __name__ == "__main__":
    main("configs/baseline.yaml")
