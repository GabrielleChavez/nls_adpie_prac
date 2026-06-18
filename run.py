import csv
import os

import handwriting_tasks as hw
import eyetracking_tasks as et
import process_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ADPIE import ADPIE
from stat_analysis import kruskal_test, pairwise_mannwhitney

def main():
    # Load and process data
    print("Loading and processing data...")
    completed_data = process_data.get_completed_paths(verbose=True)


def run_analysis(completed_data, out_path="pursuit_path_results.csv"):
    """
    Compute sequence error and gaze-pen lag features for all samples
    contained in completed_data.

    Parameters
    ----------
    completed_data : dict
        {
            "AD":  [sample1, sample2, ...],
            "PD":  [...],
            "CTL": [...]
        }

    Returns
    -------
    rows : list[dict]
        Feature rows written to CSV.
    """

    gt_base_names = [
        "Pursuit_path_recollection1",
        "Pursuit_path_recollection2",
        "Pursuit_path_recollection3",
        "Pursuit_path_recollection4",
        "Pursuit_path_recollection5",
        "Pursuit_path_recollection6",
        "Pursuit_path_recollection7",
    ]

    expected_clusters_lookup = {
        task: len(set(gt["point"]))
        for task, gt in ADPIE.pursuit_path.items()
    }

    rows = []

    for class_name, samples in completed_data.items():

        print(f"\n[{class_name}] {len(samples)} samples")

        for data in samples.values():

            task_name = data.get("task")

            # Skip tasks not used in this analysis
            if task_name not in gt_base_names:
                continue

            expected_clusters = expected_clusters_lookup.get(task_name, np.nan)

            subject_id = str(
                data.get("subject_id")
                or data.get("id")
                or "unknown"
            )

            base = {
                "filename": f"{subject_id}_{task_name}",
                "ID": subject_id,
                "label": class_name,
                "task": task_name,
                "session": subject_id[-1] if len(subject_id) > 0 else None,
                "expected_clusters": expected_clusters,
            }

            # -------------------------------------------------
            # Centroid extraction
            # -------------------------------------------------
            try:
                centroids, labels, k = et.extract_centroids_spatiotemporal(data)

                noise_ratio = (
                    float(np.sum(labels == -1) / len(labels))
                    if len(labels) > 0
                    else np.nan
                )

            except Exception as e:

                print(
                    f"Centroid extraction failed "
                    f"— {subject_id}: {e}"
                )

                rows.append({
                    **base,
                    "n_clusters": 0,
                    "noise_ratio": np.nan,
                    "expected_order": None,
                    "observed_order": None,
                    "sequence_error": np.nan,
                    "normalised_seq_err": np.nan,
                    "n_expected": np.nan,
                    "n_observed": np.nan,
                    "mean_lag_s": np.nan,
                    "std_lag_s": np.nan,
                    "n_valid_lag": np.nan,
                    "n_eye_leads": np.nan,
                    "n_pen_leads": np.nan,
                    "reversed_ratio": np.nan,
                    "error": str(e),
                })

                continue

            # -------------------------------------------------
            # Centroid matching
            # -------------------------------------------------
            try:

                centroid_error = et.compute_centroid_error(
                    centroids,
                    et._grid,
                    task_name=task_name,
                )

            except Exception as e:

                print(
                    f"Centroid matching failed "
                    f"— {subject_id}: {e}"
                )

                rows.append({
                    **base,
                    "n_clusters": k,
                    "noise_ratio": noise_ratio,
                    "expected_order": None,
                    "observed_order": None,
                    "sequence_error": np.nan,
                    "normalised_seq_err": np.nan,
                    "n_expected": np.nan,
                    "n_observed": np.nan,
                    "mean_lag_s": np.nan,
                    "std_lag_s": np.nan,
                    "n_valid_lag": np.nan,
                    "n_eye_leads": np.nan,
                    "n_pen_leads": np.nan,
                    "reversed_ratio": np.nan,
                    "error": str(e),
                })

                continue

            # -------------------------------------------------
            # Sequence error
            # -------------------------------------------------
            seq = {}

            try:

                seq = et.compute_sequence_error(
                    centroid_error,
                    task_name,
                )

            except Exception as e:

                print(
                    f"Sequence error failed "
                    f"— {subject_id}: {e}"
                )

            # -------------------------------------------------
            # Gaze-pen lag
            # -------------------------------------------------
            lag_summary = {}

            try:

                lag_results = et.compute_gaze_pen_lag(
                    centroid_error,
                    data,
                    task_name,
                    et._grid,
                )

                lag_summary = et.summarise_lag(lag_results)

            except Exception as e:

                print(
                    f"Gaze-pen lag failed "
                    f"— {subject_id}: {e}"
                )

            # -------------------------------------------------
            # Save row
            # -------------------------------------------------
            rows.append({
                **base,
                "n_clusters": k,
                "noise_ratio": round(noise_ratio, 4),

                "expected_order":
                    str(seq.get("expected_order", [])),

                "observed_order":
                    str(seq.get("observed_order", [])),

                "sequence_error":
                    seq.get("sequence_error", np.nan),

                "normalised_seq_err":
                    round(
                        seq.get("normalised_error", np.nan),
                        4,
                    ),

                "n_expected":
                    seq.get("n_expected", np.nan),

                "n_observed":
                    seq.get("n_observed", np.nan),

                "mean_lag_s":
                    round(
                        lag_summary.get(
                            "mean_lag_s",
                            np.nan,
                        ),
                        4,
                    ),

                "std_lag_s":
                    round(
                        lag_summary.get(
                            "std_lag_s",
                            np.nan,
                        ),
                        4,
                    ),

                "n_valid_lag":
                    lag_summary.get(
                        "n_valid",
                        np.nan,
                    ),

                "n_eye_leads":
                    lag_summary.get(
                        "n_eye_leads",
                        np.nan,
                    ),

                "n_pen_leads":
                    lag_summary.get(
                        "n_pen_leads",
                        np.nan,
                    ),

                "reversed_ratio":
                    round(
                        lag_summary.get(
                            "reversed_ratio",
                            np.nan,
                        ),
                        4,
                    ),

                "error": None,
            })

    # ---------------------------------------------------------
    # Write CSV
    # ---------------------------------------------------------

    fieldnames = [
        "filename",
        "ID",
        "label",
        "task",
        "session",
        "n_clusters",
        "noise_ratio",
        "expected_clusters",
        "expected_order",
        "observed_order",
        "sequence_error",
        "normalised_seq_err",
        "n_expected",
        "n_observed",
        "mean_lag_s",
        "std_lag_s",
        "n_valid_lag",
        "n_eye_leads",
        "n_pen_leads",
        "reversed_ratio",
        "error",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nSaved {len(rows)} rows "
        f"→ {os.path.abspath(out_path)}"
    )

    df = pd.DataFrame(rows)
    return df
 

if __name__ == "__main__":
    completed_data = process_data.get_completed_paths(verbose=True)
    df = run_analysis(completed_data)

    features = [
        "sequence_error",
        "normalised_seq_err",
        "mean_lag_s",
        "noise_ratio",
        "reversed_ratio",
        "n_observed",
        "mean_lag_s",
        "std_lag_s",
    ]

    for feature in features:

        result = kruskal_test(df, feature)

        print(
            f"{feature:<25}"
            f"H={result['H']:.3f} "
            f"p={result['p']:.4f}"
        )

        print(pairwise_mannwhitney(df, feature))
        print()
    