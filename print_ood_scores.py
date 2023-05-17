import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, help="Path to results file.")
    args = parser.parse_args()
    return args


def print_ood_scores(results_file):
    results = pd.read_csv(results_file)
    datasets = results["filename"].apply(lambda x: x.split("_")[0]).unique()
    out_datasets = list(datasets)
    out_datasets.remove("BRATS")
    results_in = results.loc[results["filename"].str.contains("BRATS")]
    for out_dataset in out_datasets:
        results_out = results.loc[results["filename"].str.contains(out_dataset)]
        # get in likelihoods and class id
        likelihoods = list(results_in["likelihood"])
        id = [1] * len(results_in["likelihood"])
        # add out likelihoods
        likelihoods.extend(list(results_out["likelihood"]))
        id.extend([0] * len(results_out["likelihood"]))
        auc = roc_auc_score(id, likelihoods)
        print(f"{out_dataset}: {auc* 100:.1f}")


if __name__ == "__main__":
    args = parse_args()
    print_ood_scores(results_file=args.results_file)
