import sys, os, subprocess, re, io, cv2, math
import numpy as np
import pandas as pd

def main():
    file_path = "/projects/NLS_ADPIE/data/ADPIE/metadata/0.metadata.csv"
    df = pd.read_csv(file_path)

    # columns
    # col = ['subject', 'label', 'diagnosis_1', 'diagnosis_2', 'age', 'sex', 'race',
    #    'ethnicity', 'years_of_diagnosis', 'years_since_symptom_onset',
    #    'glasses', 'yearsEducation', 'avg_cdr', 'sum_of_boxes_cdr_score',
    #    'moca', 'moca_visuospatial_executive', 'moca_naming', 'moca_attention',
    #    'moca_language', 'moca_abstraction', 'moca_delayed_recall',
    #    'moca_orientation', 'h&y', 'UPDRS3', 'certainty',
    #    'moca_30_days_follow_up']

    # print(df["label"].unique()) 
    # labels = ['AD', 'CTL', 'MCI_AD', 'PDM', 'MCI', 'UKN', 'ATX', 'PD']

    # count = lambda label: len(df[df["label"] == label])
    # for label in labels:
    #     print(label, ": ", count(label))

    calculate_moca_score_per_group(df)
    return

def avg_moca_score(row, moca_cols = None):
    if moca_cols is None:
        moca_cols = ['moca_visuospatial_executive', 'moca_naming', 'moca_attention',
                 'moca_language', 'moca_abstraction', 'moca_delayed_recall',
                 'moca_orientation']
    return row[moca_cols].mean()

def calculate_moca_score_per_group(df):
    # df should be the filtered dataframe with only the relevant columns
    labels = ['AD', 'CTL', 'MCI_AD', 'PDM', 'MCI','PD']
    moca_cols =  ['moca_visuospatial_executive', 'moca_naming', 'moca_attention',
                 'moca_language', 'moca_abstraction', 'moca_delayed_recall',
                 'moca_orientation'] 
    
    results = {}
    table_rows = []

    for label in labels:
        df_filtered = df[df["label"] == label].copy()
        print(f"\n{label} group: {len(df_filtered)} subjects")

        # Convert MoCA columns to numeric, coercing errors to NaN
        for col in moca_cols:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
    
        # Store row for summary table
        row_data = {"label": label}

        for moca in moca_cols:
            mean_score = df_filtered[moca].mean() # error occurs here
            row_data[moca] = round(mean_score, 2)

        # Optional overall MoCA average
        row_data["overall_avg_moca"] = round(df_filtered[moca_cols].mean(axis=1).mean(), 2)

        table_rows.append(row_data)

        # Store per-subject MoCA averages
        results[label] = df_filtered.apply(lambda row: avg_moca_score(row), axis=1)

    # Create and print summary table
    summary_df = pd.DataFrame(table_rows)

    print("\nAverage MoCA Scores Per Group")
    print(summary_df.to_string(index=False))

    return results, summary_df




if __name__ == "__main__":
    main()


