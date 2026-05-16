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

    print(df["label"].unique())

if __name__ == "__main__":
    main()


