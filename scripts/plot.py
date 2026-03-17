import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))


def parse_aug_type(run_name):
    """resnet18_basic_seed42 -> basic"""
    return run_name.split("_seed")[0].split("_", 1)[1]


def load_results(results_dir):
    res = dict()
    for subdir in os.listdir(results_dir):
        subdir_path = os.path.join(results_dir, subdir)
        if os.path.isdir(subdir_path):
            metrics_path = os.path.join(subdir_path, "metrics.json")
            corruption_metrics_path = os.path.join(subdir_path, "corruption_metrics.csv")
            if os.path.isfile(metrics_path) and os.path.isfile(corruption_metrics_path):
                with open(metrics_path, "r") as f:
                    data = json.load(f)
                data["corruption_metrics"] = pd.read_csv(corruption_metrics_path)
                aug_type = parse_aug_type(subdir)
                if aug_type not in res:
                    res[aug_type] = []
                res[aug_type].append(data)
    return res


def aggregate(results):
    """aug_type별 여러 seed의 결과를 평균"""
    agg = dict()
    for aug_type, runs in results.items():
        combined = pd.concat([r["corruption_metrics"] for r in runs])
        agg[aug_type] = {
            "mce": round(sum(r["mce"] for r in runs) / len(runs), 4),
            "clean_acc": round(sum(r["clean_acc"] for r in runs) / len(runs), 4),
            "corruption_mean_acc": round(sum(r["corruption_mean_acc"] for r in runs) / len(runs), 4),
            "corruption_metrics": combined.groupby(["corruption", "severity"])["acc"].mean().reset_index(),
        }
    return agg


def plot_mce_bar(agg, save_dir):
    """aug_type별 mCE 막대그래프 (mCE 오름차순 정렬)"""
    items = sorted(agg.items(), key=lambda x: x[1]["mce"])
    aug_types = [k for k, _ in items]
    mces = [v["mce"] for _, v in items]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(aug_types, mces, color="skyblue")
    plt.ylabel("mCE")
    plt.xlabel("Augmentation Type")
    plt.title("mCE by Augmentation Type")
    plt.xticks(rotation=30)
    for bar, mce in zip(bars, mces):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{mce:.3f}", ha="center", va="bottom")
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mce_bar.png"))
    plt.close()


def plot_severity_line(agg, save_dir):
    """aug_type별 severity별 평균 accuracy 라인 플롯"""
    plt.figure(figsize=(8, 5))
    for aug_type, data in agg.items():
        df = data["corruption_metrics"]
        mean_acc = df.groupby("severity")["acc"].mean()
        plt.plot(mean_acc.index, mean_acc.values, marker="o", label=aug_type)
    plt.xlabel("Severity")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Severity by Augmentation Type")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "severity_line.png"))
    plt.close()


def plot_corruption_heatmap(agg, save_dir):
    """aug_type x corruption 평균 accuracy 히트맵"""
    aug_types = sorted(agg.keys())
    corruption_types = sorted(
        agg[aug_types[0]]["corruption_metrics"]["corruption"].unique()
    )

    acc_matrix = []
    for aug_type in aug_types:
        df = agg[aug_type]["corruption_metrics"]
        row = [
            df[df["corruption"] == c]["acc"].mean() if c in df["corruption"].values else float("nan")
            for c in corruption_types
        ]
        acc_matrix.append(row)

    acc_df = pd.DataFrame(acc_matrix, index=aug_types, columns=corruption_types)
    plt.figure(figsize=(1.5 * len(corruption_types), 0.6 * len(aug_types) + 3))
    im = plt.imshow(acc_df, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Accuracy")
    plt.xticks(range(len(corruption_types)), corruption_types, rotation=45, ha="right")
    plt.yticks(range(len(aug_types)), aug_types)
    plt.xlabel("Corruption Type")
    plt.ylabel("Augmentation Type")
    plt.title("Accuracy Heatmap: Augmentation Type x Corruption Type")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "corruption_heatmap.png"))
    plt.close()


def main(results_dir):
    results = load_results(results_dir)
    agg = aggregate(results)
    save_dir = os.path.join(results_dir, "plots")
    plot_mce_bar(agg, save_dir)
    plot_severity_line(agg, save_dir)
    plot_corruption_heatmap(agg, save_dir)
    print(f"Plots saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    main(args.results_dir)
