"""
Gabrielle Chavez © 2026

The code below extracts and processes data from a CSV file
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
import sys, os, subprocess, re, io, cv2, math
from PIL import Image
from tslearn.utils import to_time_series
from tslearn.metrics import dtw_path, dtw
from tslearn.preprocessing import TimeSeriesResampler
from boxplots_significance import mannwhitneyu_test
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter, label, center_of_mass
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
from math import ceil, sqrt
from handwriting_tasks import count_touches

PATH = "..\handwriting"
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), PATH)))
from normalize_align import normalize_handwriting_df, align_handwriting_with_eye_df, Columns, DeviceSpec

from ADPIE import ADPIE

def getDF(hw, eye=None):
    """
    Code adapted from Yuzhe to normalize handwriting data and align it with eye-tracking data if available.
    """
    if eye is None:
        df_norm = normalize_handwriting_df(hw)
        cols=Columns(time="logged_time", x="x", y="y"),
        device=DeviceSpec(raw_x_max=30182, raw_y_max=18864, screen_width_px=2880, screen_height_px=1800) 

        data = {
            "X" :  df_norm['x_norm'],
            "Y" : df_norm['y_norm'],
            "T" : df_norm['logged_time'],
            "P" : df_norm['pressure']
        }
    else:
        merged = align_handwriting_with_eye_df(
            eye=eye,
            hw=hw,
            tolerance_seconds=0.05,         
            direction="nearest",            
            suffixes=("_eye", "_hand"),     
        )

        data = {
            "X" : merged['x_norm'],
            "Y" : merged['y_norm'],
            "T" : merged['logged_time'],
            "P" : merged['pressure'],
            "gaze_x" : merged['gaze_x'],
            "gaze_y": merged['gaze_y']
        }

    return data

def render_channel(X, Y, color, thickness=1.5, upsample=4, pressure=None, use_pressure=False):
    """
    Render strokes with optional per-point colors (Nx3).
    Supports anti-aliased line rasterization.
    """
    # Remove NaNs
    # valid = np.isfinite(X) & np.isfinite(Y)
    # X, Y = X[valid], Y[valid]
    # if pressure is not None:
    #     pressure = pressure[valid]
    # if isinstance(color, np.ndarray):
    #     color = color[valid]

    # Canvas setup
    base_size = 128
    hi_size = base_size * upsample
    hi_img = np.zeros((hi_size, hi_size, 3), np.float32)

    # Normalize [-1,1] -> [0, hi_size-1]
    x_pix = ((X + 1) * 0.5 * (hi_size - 1)).astype(np.int32)
    y_pix = ((1 - (Y + 1) * 0.5) * (hi_size - 1)).astype(np.int32)  # flip y

    # Draw line segments
    for i in range(len(x_pix) - 1):
        if use_pressure and (pressure[i] <= 0 or pressure[i + 1] <= 0):
            continue
        
        thick = round(pressure[i], 1) if use_pressure else thickness
        c = color[i] if isinstance(color, np.ndarray) else color
        pt1 = (x_pix[i], y_pix[i])
        pt2 = (x_pix[i + 1], y_pix[i + 1])
        cv2.line(
            hi_img,
            pt1,
            pt2,
            color=tuple(map(float, c)),  # ensure float3
            thickness=int(round(thick * upsample)),
            lineType=cv2.LINE_AA,
        )

    # Downsample to 256×256
    img = cv2.resize(hi_img, (base_size, base_size), interpolation=cv2.INTER_AREA)
    return np.clip(img, 0, 1)

def xy2img(df, task=None, line_width=3):
    """
    Render red (subject) and green (target) strokes as an AA image.
    """
    Xr = df["X"].astype(float).values
    Yr = df["Y"].astype(float).values
    Pr = df["P"].astype(float).values
    Tr = df["T"].astype(float).values

    mask = Pr > 0

    # Normalize pressure [0, 1], threshold pen-down
    P_NORM = 3 + 10 * (Pr / ADPIE.PRESSURE_MAX)**2 
    P_NORM[mask] = 0

    # Normalize time [0, 1]
    Tr_norm = (Tr - Tr.min()) / (Tr.max() - Tr.min())
    cmap = plt.get_cmap("Reds")
    colors = cmap(Tr_norm)[:, :3]

    #  Render red channel (subject stroke) 
    #img_red = render_channel(Xr, Yr, color=colors, thickness=1.5, pressure=P_NORM, use_pressure=True)

    #  Render green channel (subject stroke) 
    
    Xr = Xr[mask]
    Yr = Yr[mask]
    img_green = render_channel(Xr, Yr, color=(0, 1, 0), thickness=line_width, pressure=P_NORM)

    #  Render blue channel (pursuit path) 
    path = ADPIE.pursuit_path[task]
    Xg = np.array(path["x"])
    Yg = np.array(path["y"])
    img_blue = render_channel(Xg, Yg, color=(0, 0, 1), thickness=line_width)

    # Combine (clamp to 1)
    #img = np.clip(img_red + img_green + img_blue, 0, 1)
    img = np.clip(img_green + img_blue, 0, 1)

    return img.astype(np.float32)

def get_single_class(_labels, class_, get_img = False, set_width=3, hw_only=False, b_or_m="M", subject_id=False):
    """
    Function to load NLS data
    Parameters
    ----------
    data_name : str
        Path to the data
    Returns
    -------
    nls_data : dict
        Dictionary with keys as image names and values as images
    """
    # First Gather Disease info for each patient
    hw_directory = "/projects/NLS_ADPIE/data/ADPIE/Processed/handwriting/cleaned"
    eye_directory = "/projects/NLS_ADPIE/data/ADPIE/Processed/eyetracking/csv/" 
    file_path = "/projects/NLS_ADPIE/data/ADPIE/metadata/0.metadata.csv"

    df_og = pd.read_csv(file_path)
    # Goal is to only extract PD and control
    CLASS_1 = class_
    valid_labels = rf"{CLASS_1}"

    if _labels == "mole":
        _labels = "Pursuit_path_recollection"

    if CLASS_1 == 'all_ad':
        df_og = df_og[(df_og['label'] == 'AD') | (df_og['label'] == 'MCI_AD') | (df_og['label'] == 'MCI')]
        valid_labels = rf"AD|MCI_AD|MCI|"

    else:
        df_og = df_og[(df_og['label'] == CLASS_1)]

    # Create Dictionary
    ID_TYPE = pd.Series(df_og['label'].values, index=df_og['subject']).to_dict()

    # Second Iterate through folders to get svc files
    class_1_data = []
    count = 0
    for folder in sorted(os.listdir(hw_directory)): # Iterate through folders
        if folder in ID_TYPE.keys() and (ID_TYPE[folder] == valid_labels):

            folder_path = os.path.join(hw_directory, folder)
            taskType = ID_TYPE[folder] 
            if ("NLS_296" in folder_path or "NLS_295" in folder_path or "NLS_265" in folder_path):
                continue

            for svc_file in sorted(os.listdir(folder_path)): 
                task = "_".join(svc_file.split("_")[3:])[:-4]
                if re.search(_labels, svc_file) and taskType in CLASS_1:
                   
                    svc_path_hw = os.path.join(folder_path, svc_file)
                    subject_fm = "_".join(svc_file.split("_")[:3])
                    if hw_only:
                        df = getDF(svc_path_hw)
                    else:
                        try:
                            
                            svc_path_eye = eye_directory +'/' + folder + "/" + subject_fm + "_" + b_or_m.upper() + "_" + task + ".csv"
                            df = getDF(svc_path_hw, svc_path_eye)
                        except Exception as e:
                            print(e)
                            print(subject_fm, "task: ", task)
                            continue

                    # if count == 34:
                    #     print(svc_path_hw)
                    #     print(svc_path_eye)
                    # count += 1

                    P_raw = df["P"].astype(float).values
                    T_raw = df["T"].astype(float).values
                    X_raw = df["X"].astype(float).values
                    Y_raw = df["Y"].astype(float).values

                    # normalize pressure
                    P_norm = 3 + 6 * (P_raw / ADPIE.PRESSURE_MAX) ** 2
                    P_norm[P_raw <= 0] = 0

                    # valid writing points
                    mask = P_raw > 0

                    # normalize itme
                    T_norm = T_raw - np.min(T_raw)

                    # apply mask
                    x = X_raw[mask]
                    y = Y_raw[mask]
                    P_norm = P_norm[mask]

                    #generate image
                    img = xy2img(df, task, set_width)

    
                    moca_cols = ['moca', 'moca_visuospatial_executive', 'moca_attention', 'moca_delayed_recall', 'moca_orientation']
                    session = int(svc_file.split("_")[1][:-2]) - 1

                    # Filter to this specific subject
                    subject_row = df_og[df_og['subject'] == folder]  # 'folder' is the subject ID in your loop
                    moca_dict = {}
                    for moca in moca_cols:
                        try:
                            moca_raw = str(subject_row[moca].values[0])
                            moca_sessions = moca_raw.split(";")
                            moca_dict[moca] = float(moca_sessions[session]) if session < len(moca_sessions) else moca_sessions[-1]
                        except Exception as e:
                            print(f"Error processing MoCA column {moca} for subject {folder}: {e}")
                            moca_dict[moca] = np.nan  # or some default value

                    # build dict
                    sample = {
                        "img": img,
                        "X": x,
                        "Y": y,
                        "P": P_norm,
                        "T": T_norm,
                        "task": task,
                        "moca" : moca_dict
                    }

                    #include eyetracking data
                    if not hw_only:
                        gaze_x = df["gaze_x"].astype(float).values[mask]
                        gaze_y = df["gaze_y"].astype(float).values[mask]

                        sample["gazeX"] = gaze_x
                        sample["gazeY"] = gaze_y
                    
                    if subject_id:
                        sample["subject_id"] = subject_fm

                    # Store Sample
                    class_1_data.append(sample)
            
    return class_1_data

def _get_completed_paths(data):
    # data is a dictionary 
    completed_paths = {}

    for participant in data:
        _, vertex_accuracy = count_touches(participant, returnAll=False )

        if math.isclose(vertex_accuracy, 1.0):
            completed_paths[participant['subject_id']] = participant

    return completed_paths

def get_completed_paths(task = "mole", verbose=False):
    """
    This function only extracts the paths that were complete. I.E the individual touched all of the vertices in the pattern
    Parameters
    ----------
    task : str
        The task to extract completed paths for (e.g., "mole", "spiral
    verbose : bool
        Whether to print detailed information during processing 
    Returns
    -------
    completed_data : dict
        Dictionary with keys as group names and values as lists of completed path samples
        It is in the format:
        completed_data = {"AD": [sample1, sample2, ...], "PD": [...], "CTL": [...], ...}
        where each sample is a dictionary containing the following keys:
        - "img": The rendered image of the handwriting sample
        - "X": The x-coordinates of the handwriting sample
        - "Y": The y-coordinates of the handwriting sample
        - "P": The normalized pressure values of the handwriting sample
        - "T": The normalized time values of the handwriting sample
        - "task": The task name (e.g., "mole", "spiral")
        - "moca": A dictionary containing MoCA scores for the participant
        - "gazeX": The x-coordinates of the gaze data (if available)
        - "gazeY": The y-coordinates of the gaze data (if available)
        - "subject_id": The subject ID (if available)

    """
    nd_groups = ["AD", "PD", "CTL"]

    mci_ad = get_single_class(task, "MCI_AD", hw_only=False, subject_id=True)
    mci = get_single_class(task, "MCI", hw_only=False, subject_id=True)
    ad = get_single_class(task, "AD", hw_only=False, subject_id=True)
    pdn = get_single_class(task, "PD", hw_only=False, subject_id=True)
    ctl = get_single_class(task, "CTL", hw_only=False, subject_id=True)

    nd_data = [mci_ad, mci, ad, pdn, ctl]
    nd_data_name = ["MCI_AD", "MCI", "AD", "PD", "CTL"]

    completed_data = {}

    for (group, name) in zip(nd_data, nd_data_name):
        if verbose:
            print(f"Processing group: {name}")
            print(f"Total participants in {name}: {len(group)}")
        completed_data[name] = _get_completed_paths(group)

    if verbose:
        print("\nSummary of Completed Paths:")
        for name, paths in completed_data.items():
            print(f"{name}: {len(paths)} completed paths")

    return completed_data

