import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))


def load_results(results_dir):
    """results/ 하위 폴더에서 metrics.json을 읽어 dict 리스트로 반환"""
    metrics = dict()
    for subdir in os.listdir(results_dir):
        subdir_path = os.path.join(results_dir, subdir)
        if os.path.isdir(subdir_path):
            metrics_path = os.path.join(subdir_path, "metrics.json")
            if os.path.isfile(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics[subdir] = json.load(f)
    return metrics


def print_table(rows):
    """결과를 표 형태로 출력"""
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def save_csv(rows, save_path):
    """결과를 CSV로 저장"""
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)


def main(results_dir="results"):
    metrics = load_results(results_dir)
    # Process and display results
    rows = []
    for model_name, metric in metrics.items():
        row = {
            "Model": model_name,
            "Clean Acc": metric["clean_acc"],
            "Corruption Mean Acc": metric["corruption_mean_acc"],
            "MCE": metric["mce"]
        }
        rows.append(row)
    print_table(rows)
    save_csv(rows, os.path.join(results_dir, "comparison.csv"))


if __name__ == "__main__":
    main()
