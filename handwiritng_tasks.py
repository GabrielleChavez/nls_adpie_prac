import pandas as pd
import numpy as np
from tslearn.metrics import dtw_path, dtw
from boxplots_significance import mannwhitneyu_test
from sklearn.cluster import DBSCAN
from ADPIE import ADPIE

def count_touches(person_points, returnAll=True):
    task = person_points["task"]
    gt_points = ADPIE.pursuit_path[task]
    radius = ADPIE.radii[task]
    gx = gt_points['x']        # length K
    gy = gt_points['y']
    px = person_points['X']    # length N
    py = person_points['Y']

    # Pairwise distances (N,K)
    dx = px[:, None] - gx[None, :]
    dy = py[:, None] - gy[None, :]
    dist = np.sqrt(dx*dx + dy*dy)

    # Binary proximity matrix
    touches = dist < radius

    # Per-GT point: was it touched?
    touched_any = touches.any(axis=0)       # numpy array, shape (K,)
    touched_any_list = touched_any.tolist() # convert to list for output

    # Count touched vertices
    num_touched = int(touched_any.sum())

    vertex_accuracy = num_touched / len(gx)

    # Determine missed vertices
    if vertex_accuracy < 1:
        vertices_missed = [vertex for touched, vertex in zip(touched_any_list, gt_points['point']) if not touched]
    else:
        vertices_missed = []
    if returnAll:
        return num_touched, touched_any_list, vertex_accuracy, vertices_missed
    return num_touched, vertex_accuracy

def find_avg_per_gt(gt, data, radius=0.02033333333333333):
    avg_points = np.zeros((len(data), len(gt['x'])))
    for i, d in enumerate(data):
        avg_points[i], _ = count_touches(gt, d, radius=radius)
    
    total_touches = avg_points.sum(axis=0) / len(data)
    
    return total_touches

def dynamic_time_warp(person_points, verbose=False):
    """
    ---
    returns
     dist : (float) holds information regarding duration difference, amount of stretching, temporal distoration, and unusual pacing (best for classification from my understanding)

     normalized_dist : (float)  compares geometric similarity between GT and subject, the angerage "mismatch" between two patterns
    """

    task = person_points["task"]

    gt_points = ADPIE.pursuit_path[task]

    T1= np.stack((gt_points['x'], gt_points['y']), axis=1)
    T2 = np.stack((person_points['X'], person_points['Y']), axis=1)

    # Compute DTW path with Sakoe-Chiba constraint
    radius = int(len(T2) * 0.05)

    path, dist = dtw_path(
        T1,
        T2,
        global_constraint="sakoe_chiba",
        sakoe_chiba_radius=radius
    )

    dist = dtw(T1, T2)

    normalized_dist = dist / len(path) # disregard warping length
    
    if verbose:
        print(T1.shape, T2.shape, T1_points.shape, T2_points.shape)
        print("DTW distance:", dist)
        print("DTW average distance:", normalized_dist)
        print("Path:", path)
    
    return dist, normalized_dist

def percent_overlap(participant):
    # extract GT from 3D img ALWAYS BLUE
    img = participant['img']
    img_gt = np.zeros_like(img)
    img_gt[:, :, 2] = img[:, :, 2]

    # extract PATH from 3D img ALWAYS GREEN
    img_path = np.zeros_like(img)
    img_path[:, :, 1] = img[:, :, 1]
    
    gt_flatten = np.sum(img_gt, axis=-1)
    path_flatten = np.sum(img_path, axis=-1)

    gt_mask = gt_flatten > 0
    path_mask = path_flatten > 0

    intersection = np.sum(gt_mask & path_mask)
    union = np.sum(gt_mask | path_mask)

    if union == 0:
        return 0.0  # no strokes

    return (intersection / union) 




def extract_features_for_radius(gt, nd_data, labels, radius):
    rows = []

    for nd, label in zip(nd_data, labels):
        for subj_id, sig in enumerate(nd):
            num_touched, vertex_acc = count_touches(gt, sig, radius)

            rows.append({
                'subject': f'{label}_{subj_id}',
                'label': label,
                'vertex_accuracy': vertex_acc,
                'num_touched': num_touched
            })

    return pd.DataFrame(rows)

def filter_groups(df, g1='AD', g2='CTL'):
    return df[df['label'].isin([g1, g2])]

def grid_search(gt, nd_data, labels, radii, group_one_name="PD", group_two_name='CTL'):
    results = []
    for r in radii:
        df = extract_features_for_radius(gt, nd_data, labels, r)

        res = mannwhitneyu_test(
            df,
            cols=['vertex_accuracy'],
            group_one_name=group_one_name,
            group_two_name=group_two_name
        )

        raw_p, corr_p, sig = res['vertex_accuracy']

        results.append({
            'radius': r,
            'raw_p': raw_p,
            'corrected_p': corr_p,
            'significant': sig
        })

    results_df = pd.DataFrame(results)

    best = results_df.loc[results_df['corrected_p'].idxmin()]
    print(best)

    return results_df

def append_data(col_name, data, file_name='../mole.csv'):
    """Appends new features to spreadsheet

    Parameters
    ----------
    col_name : str
        Name of feature
    data : list
        Data related to feature
    """
    df = pd.read_csv(file_name) #Read Excel file as a DataFrame
    df[col_name] = data #add col
    
    #To save it back as Excel
    df.to_csv(file_name) #Write DateFrame back as Excel file

    return None

def run_all_features(task= "mole", file_name="pursuit_path.csv", feature_name=None, feature_funcs=None):
    if feature_name is None:
        feature_name = ["alignment_effort_DTW", "shape_similarity", "percent_overlap", "vertex_accuracy", "num_vertices_touched", "vertices_touched"]

    if feature_funcs is None: 
        feature_funcs = [dynamic_time_warp, percent_overlap, count_touches]

    nd_groups = ["AD", "PD", "CTL"]

    ad = get_single_class(task, "MCI_AD")
    pdn = get_single_class(task, "PD")
    ctl = get_single_class(task, "CTL")

    nd_data = [ad, pdn, ctl]

    for func in feature_funcs:
       
        all_results = []
        print(func.__name__)
    
        for group in nd_data:
            for participant in group:
                result = func(participant)
                all_results.append(result)

        if not(type(all_results[0]) is tuple):
            feat_col = all_results
            name = feature_name[0]
            append_data(name, feat_col, file_name=file_name)

            feature_name = feature_name[1:]

        else:
            feat_col = list(zip(*all_results))  
        
            for i, col in enumerate(feat_col):
                name = feature_name[i]
                print(name)
                append_data(name, list(col), file_name=file_name)
            try:
                feature_name = feature_name[len(feat_col):]
            except Exception as e:
                print(e)
                print(feature_name)
                feature_name = []


def completed_tasks_features(data, verbose=False):
    # the data is the full complete list of participants for a task, not separated by group
    results = {}
    for name, paths in data.items():
        if verbose: 
            print(name, paths)
        for participant in paths:
            num_touched, touched_any_list, vertex_acc, vertices_missed = count_touches(paths, participant)
            if verbose: 
                print(f"Participant {participant['subject_id']} touched {num_touched} vertices with accuracy {vertex_acc:.2f}. Missed vertices: {vertices_missed}")
            results[participant['subject_id']] = {
                'num_touched': num_touched, 
                'vertex_accuracy': vertex_acc,
                'vertices_missed': vertices_missed
            }
            break

    return results







if __name__ == "__main__":
    
    # Example usage
    # participant = {
    #     'task': 'Pursuit_path_recollection1',
    #     'X': np.array([...]),  # Participant's X coordinates
    #     'Y': np.array([...])   # Participant's Y coordinates
    # }
    
    # dist, norm_dist = dynamic_time_warp(participant)
    # print("DTW Distance:", dist)
    # print("Normalized DTW Distance:", norm_dist)



