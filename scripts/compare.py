import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))


def load_results(results_dir):
    metrics = dict()
    for subdir in os.listdir(results_dir):
        subdir_path = os.path.join(results_dir, subdir)
        if os.path.isdir(subdir_path):
            metrics_path = os.path.join(subdir_path, "metrics.json")
            if os.path.isfile(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics[subdir] = json.load(f)
    return metrics


def main(results_dir):
    metrics = load_results(results_dir)
    rows = []
    for run_name, metric in metrics.items():
        row = {
            "run": run_name,
            "clean_acc": metric["clean_acc"],
            "corruption_mean_acc": round(metric["corruption_mean_acc"], 4),
            "mce": metric["mce"],
        }
        for s in range(1, 6):
            row[f"sev{s}"] = round(metric["severity_acc"][str(s)], 4)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("mce").reset_index(drop=True)
    print(df.to_string(index=False))

    save_path = os.path.join(results_dir, "comparison.csv")
    df.to_csv(save_path, index=False)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    main(args.results_dir)
