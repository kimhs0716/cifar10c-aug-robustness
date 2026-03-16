import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))


def load_results(results_dir):
    """results/ 하위 폴더에서 metrics.json과 corruption_metrics.csv를 읽어 반환"""
    res = dict()
    for subdir in os.listdir(results_dir):
        subdir_path = os.path.join(results_dir, subdir)
        if os.path.isdir(subdir_path):
            metrics_path = os.path.join(subdir_path, "metrics.json")
            corruption_metrics_path = os.path.join(subdir_path, "corruption_metrics.csv")
            if os.path.isfile(metrics_path) and os.path.isfile(corruption_metrics_path):
                with open(metrics_path, "r") as f:
                    res[subdir] = json.load(f)
                res[subdir]["corruption_metrics"] = pd.read_csv(corruption_metrics_path)
    return res


def plot_mce_bar(results, save_dir):
    """aug_type별 mCE 막대그래프"""
    aug_types = []
    mces = []
    for aug_type, data in results.items():
        if "mce" in data:
            aug_types.append(aug_type)
            mces.append(data["mce"])
    plt.figure(figsize=(8, 5))
    bars = plt.bar(aug_types, mces, color="skyblue")
    plt.ylabel("mCE")
    plt.xlabel("Augmentation Type")
    plt.title("mCE by Augmentation Type")
    plt.xticks(rotation=30)
    for bar, mce in zip(bars, mces):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{mce:.2f}", ha='center', va='bottom')
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mce_bar.png"))
    plt.close()


def plot_severity_line(results, save_dir):
    """aug_type별 severity별 accuracy 라인 플롯"""
    plt.figure(figsize=(8, 5))
    for aug_type, data in results.items():
        if "corruption_metrics" in data:
            df = data["corruption_metrics"]
            if "severity" in df.columns and "acc" in df.columns:
                # 평균 accuracy per severity
                mean_acc = df.groupby("severity")["acc"].mean()
                plt.plot(mean_acc.index, mean_acc.values, marker='o', label=aug_type)
    plt.xlabel("Severity")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Severity by Augmentation Type")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "severity_line.png"))
    plt.close()


def plot_corruption_heatmap(results, save_dir):
    """aug_type X corruption accuracy 히트맵"""
    # Collect all corruption types
    corruption_types = set()
    for data in results.values():
        if "corruption_metrics" in data:
            corruption_types.update(data["corruption_metrics"]["corruption"].unique())
    corruption_types = sorted(list(corruption_types))
    aug_types = list(results.keys())
    # Build accuracy matrix
    acc_matrix = []
    for aug_type in aug_types:
        row = []
        df = results[aug_type].get("corruption_metrics")
        for corruption in corruption_types:
            if df is not None:
                accs = df[df["corruption"] == corruption]["acc"]
                row.append(accs.mean() if not accs.empty else float('nan'))
            else:
                row.append(float('nan'))
        acc_matrix.append(row)
    acc_df = pd.DataFrame(acc_matrix, index=aug_types, columns=corruption_types)
    plt.figure(figsize=(1.5*len(corruption_types), 0.6*len(aug_types)+3))
    im = plt.imshow(acc_df, aspect='auto', cmap='viridis')
    plt.colorbar(im, label="Accuracy")
    plt.xticks(range(len(corruption_types)), corruption_types, rotation=45, ha='right')
    plt.yticks(range(len(aug_types)), aug_types)
    plt.xlabel("Corruption Type")
    plt.ylabel("Augmentation Type")
    plt.title("Accuracy Heatmap: Augmentation Type x Corruption Type")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "corruption_heatmap.png"))
    plt.close()


def main(results_dir="results"):
    results = load_results(results_dir)
    save_dir = os.path.join(results_dir, "plots")
    plot_mce_bar(results, save_dir)
    plot_severity_line(results, save_dir)
    plot_corruption_heatmap(results, save_dir)


if __name__ == "__main__":
    main()
