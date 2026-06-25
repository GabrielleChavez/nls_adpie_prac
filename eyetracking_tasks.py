import numpy as np
import pandas as pd
from scipy.interpolate import interp1d 
import matplotlib.pyplot as plt
import os, sys, math, csv
from pathlib import Path
from PIL import Image
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..\..")))
## Now you | can import
from sklearn.cluster import KMeans, DBSCAN
from scipy.ndimage import gaussian_filter, center_of_mass
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
from math import ceil, sqrt
import process_data as pm
from ADPIE import ADPIE
from typing import List, Tuple

### GRAPHING FUNCTIONS ###
def heatmaps_from_classes(
    class_data,
    name=None,
    img_size=(512,512),
    x_range=(-1,1),
    y_range=(-1,1),
    sigma=3.0,
    normalize=True,
    cmap="hot",
    show=True,
    return_fig=False,
    save_path=None
):
    """
    Creates a single heatmap for a given participant
    """
    if not isinstance(class_data, (list, tuple)):
        raise ValueError("class_data must be a list returned by get_single_class")
    
    all_x, all_y, all_p = [], [], []

    for sample in class_data:

        x = np.asarray(sample.get("gazeX", []), dtype=float)
        y = np.asarray(sample.get("gazeY", []), dtype=float)

        if x.size == 0 or y.size == 0:
            continue

        p = sample.get("P")

        if p is None:
            p = np.ones_like(x)
        else:
            p = np.asarray(p, dtype=float)
            m = min(len(x), len(y), len(p))
            x, y, p = x[:m], y[:m], p[:m]

        all_x.append(x)
        all_y.append(y)
        all_p.append(p)

    if len(all_x) == 0:
        raise ValueError("No valid gaze points found")

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_p = np.concatenate(all_p)

    xmin, xmax = x_range
    ymin, ymax = y_range

    H, W = img_size

    x_norm = (all_x - xmin) / (xmax - xmin)
    y_norm = (all_y - ymin) / (ymax - ymin)

    px = np.clip((x_norm*(W-1)).astype(int), 0, W-1)
    py = np.clip((y_norm * (H - 1)).astype(int), 0, H - 1)

    heat = np.zeros((H,W), dtype=float)
    np.add.at(heat, (py, px), all_p)

    if sigma > 0:
        heat = gaussian_filter(heat, sigma=sigma)

    if normalize:
        mx = heat.max()
        if mx > 0:
            heat /= mx

    # CREATE FIGURE
    fig, ax = plt.subplots(figsize=(6,6))

    im = ax.imshow(
        heat,
        cmap=cmap,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax)
    )

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)

    ax.set_xlabel("gazeX")
    ax.set_ylabel("gazeY")

    if name:
        ax.set_title(name)

    fig.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)

    if show:
        plt.show()

    if return_fig:
        return fig, ax


def plot_all_subjects_grid(
    subjects_list,
    x_key="gazeX",
    y_key="gazeY",
    p_key="pressure",
    sigma=3.0,
    cmap='hot',
    img_size=(512, 512),
    normalize=True,
    title="Subject Heatmaps"
):

    n = len(subjects_list)
    cols = ceil(sqrt(n))
    rows = ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for i, (ax, data) in enumerate(zip(axes, subjects_list)):

        x = np.asarray(data.get(x_key, []), dtype=float)
        y = np.asarray(data.get(y_key, []), dtype=float)
        p = data.get(p_key)
        p = np.asarray(p if p is not None else np.ones(min(len(x), len(y))), dtype=float)

        n_pts = min(len(x), len(y), len(p))
        x, y, p = x[:n_pts], y[:n_pts], p[:n_pts]

        H, W = img_size

        if n_pts > 0:
            px = np.clip(
                ((x - x.min()) / (x.max() - x.min() + 1e-8) * (W - 1)).astype(int),
                0, W - 1
            )
            py = np.clip(
                ((1 - (y - y.min()) / (y.max() - y.min() + 1e-8)) * (H - 1)).astype(int),
                0, H - 1
            )

            heat = np.zeros((H, W), dtype=float)
            np.add.at(heat, (py, px), p)      # vectorized, no Python loop

            heat = gaussian_filter(heat, sigma=sigma)

            if normalize and heat.max() > 0:
                heat /= heat.max()
        else:
            heat = np.zeros((H, W), dtype=float)

        subject_id = data.get("subject_id") or data.get("id") or i
        ax.imshow(heat, cmap=cmap, origin='upper')
        ax.set_title(str(subject_id), fontsize=8)
        ax.axis('off')

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

    return fig


def display_class_images(
    class_data,
    img_key="img",
    title=None,
    cmap="gray",
    figsize_per_img=(3, 3)
):
    """
    Display all images from a list of sample dicts in a single figure.

    Parameters
    ----------
    class_data : list of dict
        Each dict must contain img_key mapped to a np.ndarray image.
    img_key : str
        Key to retrieve the image from each dict (default: "img").
    title : str or None
        Overall figure title.
    cmap : str
        Colormap (default: "gray"). Ignored for RGB images.
    figsize_per_img : tuple (w, h)
        Size in inches allocated per subplot.
    """

    # collect valid (index, image, subject_id) tuples
    images = []
    for i, sample in enumerate(class_data):
        img = sample.get(img_key)
        if img is None:
            print(f"  Warning: sample {i} has no '{img_key}' key, skipping.")
            continue
        img = np.asarray(img)
        subject_id = sample.get("subject_id", sample.get("id", i))
        images.append((i, img, subject_id))

    if not images:
        raise ValueError(f"No valid images found under key '{img_key}'.")

    n = len(images)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fw = figsize_per_img[0] * ncols
    fh = figsize_per_img[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh))

    # always iterate over a flat list of axes
    axes_flat = np.array(axes).flatten()

    for ax, (i, img, subject_id) in zip(axes_flat, images):
        # imshow handles (H,W), (H,W,1), (H,W,3), (H,W,4)
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze(-1)
        ax.imshow(img, cmap=cmap if img.ndim == 2 else None)
        ax.set_title(f"ID: {subject_id}", fontsize=7)
        ax.axis("off")

    # hide any unused subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.show()

    return fig


def iterate_pursuit_heatmaps(
    gt_base_names=None,
    classes=("MCI_AD", "PD", "CTL"),
    get_single_class_fn=None,
    heatmap_fn=None,
    show=True,
    save_dir=None,
    img_size=(512, 512),
    mode='gaussian',
    sigma=3.0,
    normalize=True,
    cmap='hot'
):
    """
    Displays the average heatmap for a class i.e pd, ad, or ctl for a particular path
    """

    if gt_base_names is None:
        gt_base_names = [
            "Pursuit_path_recollection1",
            "Pursuit_path_recollection2",
            "Pursuit_path_recollection3",
            "Pursuit_path_recollection4",
            "Pursuit_path_recollection5",
            "Pursuit_path_recollection6",
            "Pursuit_path_recollection7",
        ]

    if get_single_class_fn is None:
        try:
            get_single_class_fn = pm.get_single_class
        except Exception:
            raise ValueError(
                "Please provide get_single_class_fn or ensure module 'pm' is importable."
            )

    if heatmap_fn is None:
        try:
            heatmap_fn = heatmaps_from_classes
        except NameError:
            raise ValueError(
                "Please provide heatmap_fn or define heatmaps_from_classes."
            )

    # results[class][task] = figure
    results = {cls: {} for cls in classes}

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for gt_name in gt_base_names:

        print(f"Processing: {gt_name}")

        for cls in classes:

            print(f"  Loading class: {cls}")

            try:
                class_data = get_single_class_fn(gt_name, cls)

            except Exception as e:
                print(f"    Error loading {cls}: {e}")
                results[cls][gt_name] = None
                continue

            try:
                fig, ax = heatmap_fn(
                    class_data,
                    name=f"{gt_name} - {cls}",
                    img_size=img_size,
                    sigma=sigma,
                    normalize=normalize,
                    cmap=cmap,
                    show=show,
                    return_fig=True
                )

            except Exception as e:
                print(f"    Error creating heatmap: {e}")
                results[cls][gt_name] = None
                continue

            # store figure
            results[cls][gt_name] = fig

            # save if requested
            if save_dir is not None:

                out_path = os.path.join(save_dir, f"{gt_name}_{cls}.png")

                fig.savefig(out_path, bbox_inches="tight", dpi=200)

                print(f"    Saved heatmap → {out_path}")

    return results

def display_every_heatmap():
    results = iterate_pursuit_heatmaps(gt_base_names=None,
        classes=("MCI_AD", "PD", "CTL"),
        get_single_class_fn=None,
        heatmap_fn=None,
        show=False,
        save_dir=None,
        img_size=(512, 512),
        mode='gaussian',
        sigma=3.0,
        normalize=True,
        cmap='hot')
        

    rows = 3
    cols = 7

    fig, axes = plt.subplots(rows, cols, figsize=(20, 9))

    for row_idx, (cls, tasks) in enumerate(results.items()):
        for col_idx, (task, fig_obj) in enumerate(tasks.items()):
            ax = axes[row_idx, col_idx]

            if fig_obj is not None:
                # Extract the image array from the stored figure
                heat_ax = fig_obj.axes[0]          # the first axes in the figure
                im = heat_ax.images[0]             # the first image in that axes
                ax.imshow(im.get_array(), 
                        cmap="hot", 
                        origin="lower", 
                        extent=(-1, 1, -1, 1))
            else:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center")

            # Force axis limits
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.axis("off")

            if col_idx == 0:
                ax.set_ylabel(cls.replace("_", " "), fontsize=11)

            if row_idx == 0:
                ax.set_title(f"Path {col_idx+1}", fontsize=10)

    plt.suptitle("Pursuit Path Heatmaps by Class", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_spatiotemporal_clusters(
    data,
    labels,
    centroids_world,
    x_key="gazeX",
    y_key="gazeY",
    t_key="timestamps",
    title=None
):
    """
    Debug plot for extract_centroids_spatiotemporal.
    Shows:
      - Raw gaze points coloured by cluster label (noise in grey)
      - Cluster centroids marked with '+'
      - A time colorbar on a second axes to show temporal progression
    
    Parameters
    ----------
    data : dict
        Same dict passed to extract_centroids_spatiotemporal.
    labels : np.ndarray
        Cluster labels returned by extract_centroids_spatiotemporal.
    centroids_world : list of (x, y)
        Centroids returned by extract_centroids_spatiotemporal.
    """

    x = np.asarray(data.get(x_key, []), dtype=float)
    y = np.asarray(data.get(y_key, []), dtype=float)
    t = data.get(t_key)

    m = min(len(x), len(y), len(labels))
    x, y, labels = x[:m], y[:m], labels[:m]

    if t is None:
        t = np.arange(m, dtype=float)
    else:
        t = np.asarray(t, dtype=float)[:m]

    unique_labels = sorted(set(labels) - {-1})
    n_clusters    = len(unique_labels)

    # one colour per cluster from a qualitative palette
    cmap_clusters = plt.cm.get_cmap("tab10", max(n_clusters, 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    #  LEFT: points coloured by cluster 
    ax = axes[0]

    # noise first (grey, small, transparent)
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(
            x[noise_mask], y[noise_mask],
            c="lightgrey", s=8, alpha=0.4, label="noise", zorder=1
        )

    # each cluster
    for i, lab in enumerate(unique_labels):
        mask  = labels == lab
        color = cmap_clusters(i)
        ax.scatter(
            x[mask], y[mask],
            color=color, s=12, alpha=0.7, label=f"cluster {lab}", zorder=2
        )

    # centroids
    for i, (cx, cy) in enumerate(centroids_world):
        color = cmap_clusters(i)
        ax.plot(
            cx, cy, "+",
            color=color, markersize=16, markeredgewidth=2.5,
            markeredgecolor="black", zorder=3
        )
        ax.annotate(
            f"C{i}", xy=(cx, cy), xytext=(5, 5),
            textcoords="offset points", fontsize=9, fontweight="bold"
        )

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title("Clusters (colour) + centroids (+)")
    ax.legend(loc="upper right", fontsize=7, markerscale=1.5)
    ax.set_aspect("equal", adjustable="datalim")

    # RIGHT: same points coloured by time 
    ax2 = axes[1]

    # noise grey
    if noise_mask.any():
        ax2.scatter(
            x[noise_mask], y[noise_mask],
            c="lightgrey", s=8, alpha=0.3, zorder=1
        )

    # clustered points coloured by timestamp
    clustered = ~noise_mask
    sc = ax2.scatter(
        x[clustered], y[clustered],
        c=t[clustered], cmap="plasma", s=12, alpha=0.8, zorder=2
    )
    fig.colorbar(sc, ax=ax2, label="time (original units)")

    # centroids again
    for i, (cx, cy) in enumerate(centroids_world):
        ax2.plot(
            cx, cy, "+",
            color="white", markersize=16, markeredgewidth=2.5,
            markeredgecolor="black", zorder=3
        )

    ax2.set_xlabel(x_key)
    ax2.set_ylabel(y_key)
    ax2.set_title("Clustered points coloured by time")
    ax2.set_aspect("equal", adjustable="datalim")

    subject_id = data.get("subject_id") or data.get("id", "unknown")
    fig.suptitle(
        title or f"Subject: {subject_id} | clusters: {n_clusters} | noise: {noise_mask.sum()} pts",
        fontsize=13, fontweight="bold"
    )

    plt.tight_layout()
    plt.show()

    return fig

### HEATMAP FEATURES AND METRICS ###

def safe_kl_divergence(p, q):
    """Flatten heatmaps, normalize to probability distributions, compute KL divergence."""

    p = p.flatten().astype(float)
    q = q.flatten().astype(float)

    # Normalize to sum to 1 (required for entropy/KL divergence)
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)

    # Add small epsilon to avoid log(0)
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)

    return entropy(p, q)  # KL divergence D(p || q)

def get_kl_divergences():
    results = iterate_pursuit_heatmaps(gt_base_names=None,
        classes=("MCI_AD", "PD", "CTL"),
        get_single_class_fn=None,
        heatmap_fn=None,
        show=False,
        save_dir=None,
        img_size=(512, 512),
        mode='gaussian',
        sigma=3.0,
        normalize=True,
        cmap='hot')


    gt_base_names = [
                "Pursuit_path_recollection1",
                "Pursuit_path_recollection2",
                "Pursuit_path_recollection3",
                "Pursuit_path_recollection4",
                "Pursuit_path_recollection5",
                "Pursuit_path_recollection6",
                "Pursuit_path_recollection7",
            ]

            
    for path in gt_base_names:

        # Extract the heatmap array from the figure
        ad_fig  = results["MCI_AD"][path]
        pdn_fig = results["PD"][path]
        ctl_fig = results["CTL"][path]

        # Skip if any figure is None
        if ad_fig is None or pdn_fig is None or ctl_fig is None:
            print(f"{path}: Missing data, skipping KL calculation")
            continue

        # Extract the image array from the figure axes
        ad_path  = ad_fig.axes[0].images[0].get_array()
        pdn_path = pdn_fig.axes[0].images[0].get_array()
        ctl_path = ctl_fig.axes[0].images[0].get_array()

        # Compute KL divergences
        kl_ad_vs_ctl  = safe_kl_divergence(ad_path, ctl_path)
        kl_pdn_vs_ctl = safe_kl_divergence(pdn_path, ctl_path)

        print(path)
        print(f"KL Divergence AD  vs CTL: {kl_ad_vs_ctl:.4f}")
        print(f"KL Divergence PD  vs CTL: {kl_pdn_vs_ctl:.4f}")
        print()

### Cluster Groups no Time ###

def cluster_features(heatmap, threshold=0.9):
    """
    Extract cluster features from a heatmap using thresholding + connected components.

    Parameters
    ----------
    heatmap : np.ndarray
        2D normalized heatmap array.
    threshold : float
        Value in [0, 1] to binarize the heatmap. Pixels above this specific intensity are considered active. 

    Returns
    -------
    features : dict
        num_clusters      - number of distinct clusters
        cluster_sizes     - list of pixel counts per cluster
        mean_cluster_size - average cluster size
        max_cluster_size  - size of the largest cluster
        size_std          - std of cluster sizes (spread of attention)
    """

    # Binarize
    binary = (heatmap >= threshold).astype(int)

    # Label connected components
    labeled, num_clusters = label(binary)

    cluster_sizes = []
    for cluster_id in range(1, num_clusters + 1):
        size = np.sum(labeled == cluster_id)
        cluster_sizes.append(int(size))

    cluster_sizes = sorted(cluster_sizes, reverse=True)

    return {
        "num_clusters":      num_clusters,
        "cluster_sizes":     cluster_sizes,
        "mean_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
        "max_cluster_size":  cluster_sizes[0] if cluster_sizes else 0,
        "size_std":          float(np.std(cluster_sizes)) if cluster_sizes else 0.0,
    }

### Participant-level metrics ###

def get_heatmap_peak(heatmap_array, threshold=0.9):
    """
    Find the centre of mass of the hottest region in a heatmap.

    Thresholds the array to isolate the top `threshold` fraction of
    intensity values, then returns the centroid of that high-activation
    mask. Useful for identifying the dominant fixation region.

    Parameters
    ----------
    heatmap_array : np.ndarray
        2-D heatmap, typically normalised to [0, 1].
    threshold : float
        Fraction of the maximum value used to define the "peak" region.
        e.g. 0.9 keeps only pixels above 90% of the global max.

    Returns
    -------
    np.ndarray
        [x, y] pixel coordinates of the peak centroid.
    """

    # Build a binary mask of pixels that exceed threshold * global max
    mask = heatmap_array > (heatmap_array.max() * threshold)

    # center_of_mass returns (row, col) i.e. (y, x) — unpack accordingly
    y, x = center_of_mass(mask)

    return np.array([x, y])


def heatmap(
    data,
    x_key="gazeX",
    y_key="gazeY",
    p_key="pressure",
    sigma=3.0,
    img_size=(512, 512),
    normalize=True,
    show=False,
):
    """
    Build a pressure-weighted, Gaussian-smoothed gaze heatmap for one sample.

    Gaze coordinates are normalised against their own min/max so the function
    works with any coordinate range. The y-axis is flipped to convert from
    screen space (y increases downward) to image space (origin at bottom-left).

    Parameters
    ----------
    data : dict
        Single subject/sample dict containing gaze and pressure arrays.
    x_key : str
        Key for the horizontal gaze coordinate array.
    y_key : str
        Key for the vertical gaze coordinate array.
    p_key : str
        Key for the pressure / saliency weight array.
        Falls back to uniform weights (all ones) if the key is absent.
    sigma : float
        Standard deviation of the Gaussian blur in pixels.
        Larger values produce smoother, more spread-out heatmaps.
    img_size : tuple of (int, int)
        Output heatmap dimensions as (height, width) in pixels.
    normalize : bool
        If True, scale the final heatmap so its maximum value is 1.0.
    show : bool
        If True, render the heatmap with matplotlib before returning.

    Returns
    -------
    heat : np.ndarray
        2-D array of shape (H, W) containing the heatmap values.
        Always returned even when show=True.
    """

    # Extract arrays from the sample dict; default to empty / uniform if missing
    x = np.asarray(data.get(x_key, []), float)
    y = np.asarray(data.get(y_key, []), float)
    p = np.asarray(data.get(p_key, np.ones(len(x))), float)

    # Truncate all arrays to the shortest one to prevent index mismatches
    n = min(len(x), len(y), len(p))
    x, y, p = x[:n], y[:n], p[:n]

    H, W = img_size
    heat = np.zeros((H, W))

    if n > 0:
        # Normalise x to [0, 1] against its own range; 1e-8 avoids division by
        # zero when all x values are identical
        px = np.clip(
            ((x - x.min()) / (x.max() - x.min() + 1e-8) * (W - 1)).astype(int),
            0, W - 1
        )

        # Normalise y and flip vertically: screen y=0 is top, image y=0 is bottom
        py = np.clip(
            ((1 - (y - y.min()) / (y.max() - y.min() + 1e-8)) * (H - 1)).astype(int),
            0, H - 1
        )

        # Accumulate pressure weights at each pixel (handles duplicate indices
        # correctly, unlike heat[py, px] += p which would miss repeated coords)
        np.add.at(heat, (py, px), p)

        # Smooth the sparse point accumulation into a continuous density field
        heat = gaussian_filter(heat, sigma)

        if normalize and heat.max() > 0:
            heat /= heat.max()

    if show:
        plt.imshow(heat, cmap="hot", origin="lower")
        plt.title("Patient Heatmap")
        plt.axis("off")
        plt.show()

    return heat

def compare_patient_to_groups(data, task_name, group_heatmaps):
    """
    Compare an individual patient's gaze heatmap against each group's
    average heatmap for a given task.

    Two metrics are computed per group:
    - Peak distance  : Euclidean distance between the individual's peak
                       fixation location and the group's peak fixation location.
    - KL divergence  : How much the individual's gaze distribution diverges
                       from the group's distribution. A value near 0 means the
                       patient's gaze pattern closely resembles the group.

    Parameters
    ----------
    data : dict
        Single patient sample dict containing gaze arrays, passed directly
        to heatmap().
    task_name : str
        Key used to look up the correct heatmap in each group's task dict
        e.g. "Pursuit_path_recollection1".
    group_heatmaps : dict
        Nested dict of the form {group_label: {task_name: np.ndarray}},
        where each array is a pre-computed group-average heatmap.
        e.g. {"MCI_AD": {"task1": arr}, "PD": {"task1": arr}, ...}

    Returns
    -------
    comparisons : dict
        Dict keyed by group label, each value containing:
            "peak_distance" : float
                Pixel distance between the two peak centroids.
            "kl_divergence" : float
                KL divergence from the individual to the group distribution.
    """

    # Build the individual's heatmap and locate their dominant fixation region
    individual_heat = heatmap(data)
    ind_peak = get_heatmap_peak(individual_heat)

    comparisons = {}

    for group_label, tasks in group_heatmaps.items():

        # Retrieve the pre-computed average heatmap for this group and task
        group_heat = tasks[task_name]

        # Find where the group collectively fixates most
        group_peak = get_heatmap_peak(group_heat)

        # Spatial metric: how far apart are the two peak fixation locations
        dist = euclidean(ind_peak, group_peak)

        # Distribution metric: how different is the individual's overall gaze
        # pattern from the group's (treats heatmaps as probability distributions)
        kl = safe_kl_divergence(individual_heat, group_heat)

        comparisons[group_label] = {
            "peak_distance": dist,
            "kl_divergence": kl
        }

    return comparisons

def extract_centroids_spatiotemporal(
    data,
    eps=0.05,
    min_samples=5,
    beta=0.2,
    x_key="gazeX",
    y_key="gazeY",
    p_key="P",
    t_key="T"
):
    """
    Extract spatio-temporal cluster centroids using DBSCAN with a combined
    spatial + temporal distance metric.

    Parameters
    ----------
    data : dict
        Single subject/sample dict with gaze and timestamp arrays.
    eps : float
        DBSCAN neighborhood radius in normalized (x, y, t) space.
        Typical starting range: 0.02 – 0.08.
    min_samples : int
        Minimum points to form a dense region (core point).
    beta : float
        Time weight relative to spatial axes (both normalized to 0..1).
        beta=0.0  → purely spatial clustering
        beta=0.2  → time counts as 20% of a spatial axis
        beta=1.0  → time weighted equally to spatial axes
    x_key, y_key : str
        Keys for gaze coordinate arrays in data dict.
    p_key : str
        Key for pressure / weight array (used to compute weighted centroids).
    t_key : str
        Key for timestamp array (seconds, ms, or frame index — just be consistent).

    Returns
    -------
    centroids_world : list of (x, y) tuples
        Weighted centroids per cluster in original coordinate space.
        Noise points (DBSCAN label -1) are excluded.
    labels : np.ndarray
        DBSCAN cluster label per gaze point (-1 = noise).
    n_clusters : int
        Number of clusters found (excluding noise).
    """

    # 1. Load arrays 
    x = np.asarray(data.get(x_key, []), dtype=float)
    y = np.asarray(data.get(y_key, []), dtype=float)
    p = data.get(p_key)
    t = data.get(t_key)

    m = min(len(x), len(y))
    if m == 0:
        raise ValueError("No gaze samples found in data.")

    x, y = x[:m], y[:m]

    # pressure — fallback to uniform weights
    if p is None:
        print("ERROR")
        return

    # timestamps — fallback to frame indices
    if t is None:
        print("  Warning: no timestamps found, using frame indices as time.")
        t = np.arange(m, dtype=float)
    else:
        t = np.asarray(t, dtype=float)[:m]

    # 2. Normalize all axes to [0, 1]
    safe_norm = lambda arr: (arr - arr.min()) / (arr.max() - arr.min()) if (arr.max() - arr.min()) > 0 else np.zeros_like(arr)

    x_norm = safe_norm(x)
    y_norm = safe_norm(y)
    t_norm = safe_norm(t)


def extract_centroids_spatiotemporal(
    data,
    eps=0.05,
    min_samples=5,
    beta=0.2,
    x_key="gazeX",
    y_key="gazeY",
    p_key="P",
    t_key="T"
):
    """
    Extract spatio-temporal cluster centroids for eye movement data using DBSCAN.
    
    EM-specific behavior:
    - Only include points where P > 0
    - Centroids are computed as simple mean (not weighted by P)
    
    Parameters
    ----------
    data : dict
        Single subject/sample dict with gaze coordinates, timestamps, and pressure array.
    eps : float
        DBSCAN neighborhood radius in normalized (x, y, t) space.
    min_samples : int
        Minimum points to form a dense region (core point).
    beta : float
        Temporal scaling factor (time weight relative to spatial axes)
    x_key, y_key : str
        Keys for gaze coordinates
    p_key : str
        Key for pressure (used only for filtering, not weighting)
    t_key : str
        Key for timestamps
    
    Returns
    -------
    centroids_world : list of (x, y) tuples
        Spatial centroids of clusters in original coordinate space.
    labels : np.ndarray
        Cluster labels per gaze point (-1 = noise)
    n_clusters : int
        Number of clusters found (excluding noise)
    """
    
    # 1. Load arrays
    x = np.asarray(data.get(x_key, []), dtype=float)
    y = np.asarray(data.get(y_key, []), dtype=float)
    p = data.get(p_key)
    t = data.get(t_key)
    
    m = min(len(x), len(y))
    if m == 0:
        raise ValueError("No gaze samples found in data.")
    
    x, y = x[:m], y[:m]
    
    # 2. Filter points where P > 0

    if p is None:
        # if P not provided, assume all points valid
        mask_valid = np.ones_like(x, dtype=bool)
    else:
        p = np.asarray(p, dtype=float)[:m]
        mask_valid = p > 0
    
    x = x[mask_valid]
    y = y[mask_valid]
    
    if t is None:
        t = np.arange(len(x), dtype=float)
    else:
        t = np.asarray(t, dtype=float)[:m]
        t = t[mask_valid]
    
    if len(x) == 0:
        # all points were filtered out
        return [], np.array([]), 0
    
    # 3. Normalize coordinates and time to [0, 1]
    def safe_norm(arr):
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / rng if rng > 0 else np.zeros_like(arr)
    
    x_norm = safe_norm(x)
    y_norm = safe_norm(y)
    t_norm = safe_norm(t)
    
    # 4. Build feature matrix (x, y, beta * t)
    coords = np.column_stack([x_norm, y_norm, beta * t_norm])
    
    # 5. Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(coords)
    
    unique_labels = sorted(set(labels) - {-1})
    n_clusters = len(unique_labels)
    
    if n_clusters == 0:
        return [], labels, 0
    
    # 6. Compute simple (unweighted) centroids
    centroids_world = []
    for lab in unique_labels:
        mask = labels == lab
        wx = x[mask].mean()
        wy = y[mask].mean()
        centroids_world.append((wx, wy))
    
    return centroids_world, labels, n_clusters


def extract_features_for_dbscan(nd_data, labels, eps, min_samples, beta):
    """
    Extract cluster-based features for all subjects using extract_centroids_spatiotemporal.

    Returns a DataFrame with features per subject.
    """
    rows = []
    
    for nd, label in zip(nd_data, labels):
        for subj_id, sig in enumerate(nd):
            try:
                centroids, cluster_labels, n_clusters = extract_centroids_spatiotemporal(
                    sig,
                    eps=eps,
                    min_samples=min_samples,
                    beta=beta
                )
            except Exception as e:
                # skip subjects with errors
                print(e)
                continue
            
            # Example features: number of clusters and max cluster size
            cluster_sizes = [np.sum(cluster_labels == c) for c in set(cluster_labels) if c != -1]
            max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
            mean_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0

            rows.append({
                'subject': f'{label}_{subj_id}',
                'label': label,
                'n_clusters': n_clusters,
                'max_cluster_size': max_cluster_size,
                'mean_cluster_size': mean_cluster_size
            })
    
    return pd.DataFrame(rows)


def mannwhitneyu_test(df, col, group_one_name='PD', group_two_name='CTL'):
    """
    Simple wrapper to compute Mann-Whitney U test between two groups for one column.
    """
    g1 = df[df['label'] == group_one_name][col].values
    g2 = df[df['label'] == group_two_name][col].values

    if len(g1) == 0 or len(g2) == 0:
        return {col: (None, None, False)}
    
    stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
    corrected_p = p  # placeholder if multiple testing correction needed
    sig = corrected_p < 0.05
    
    return {col: (p, corrected_p, sig)}


def grid_search_dbscan(task, nd_data, labels, eps_values, min_samples_values, beta_values,
                       group_one_name="PD", group_two_name='CTL', feature_col='n_clusters'):
    """
    Grid search over eps, min_samples, beta for extract_centroids_spatiotemporal.
    Returns results DataFrame and prints best parameter combination.
    """
    results = []

    for eps in eps_values:
        for min_samples in min_samples_values:
            for beta in beta_values:
                df = extract_features_for_dbscan(nd_data, labels, eps, min_samples, beta)
                res = mannwhitneyu_test(df, feature_col, group_one_name, group_two_name)
                
                raw_p, corrected_p, sig = res[feature_col]

                results.append({
                    'task': task,
                    'eps': eps,
                    'min_samples': min_samples,
                    'beta': beta,
                    'raw_p': raw_p,
                    'corrected_p': corrected_p,
                    'significant': sig,
                    'label': labels
                })

    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['corrected_p'].idxmin()]
    print("Best parameter combination:")
    print(best)
    
    return results_df


GRID_COORDS = [
(-0.736,  0.613333),
(-0.368,  0.613333),
( 0.000,  0.613333),
( 0.368,  0.613333),
( 0.736,  0.613333),
(-0.736,  0.000000),
(-0.368,  0.000000),
( 0.000,  0.000000),
( 0.368,  0.000000),
( 0.736,  0.000000),
(-0.736, -0.613333),
(-0.368, -0.613333),
( 0.000, -0.613333),
( 0.368, -0.613333),
( 0.736, -0.613333),
]

_grid = np.array(GRID_COORDS)  # shape (15,2)


def compute_velocity_arrays(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous gaze velocity in normalised units/second
    directly from numpy arrays.

    Parameters
    ----------
    t : np.ndarray
        Timestamp array in seconds.
    x, y : np.ndarray
        Gaze coordinate arrays in normalised units.

    Returns
    -------
    np.ndarray
        Velocity array, same length as t. First sample is padded
        with the first computed interval velocity.
    """
    dt = np.diff(t)
    dx = np.diff(x)
    dy = np.diff(y)

    vel_between = np.sqrt((dx / (dt+ 1e-9))**2 + (dy / (dt+ 1e-9))**2) 

    # Pad first sample by repeating the first computed velocity
    return np.concatenate(([vel_between[0]], vel_between))

def angle_between(a, b):
    """
    Compute the unsigned angle in radians between two 2-D vectors.

    Uses the dot-product formula with clipping to guard against floating-point
    values slightly outside [-1, 1] that would make arccos undefined.
    Returns NaN if either vector is zero-length (undefined direction).

    Parameters
    ----------
    a, b : array-like of shape (2,)
        Input vectors in normalised coordinate space.

    Returns
    -------
    float
        Angle in radians in [0, π], or np.nan if either vector is degenerate.
    """
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)

    # A zero-length vector has no direction — angle is undefined
    if an == 0 or bn == 0:
        return np.nan

    # Clip to [-1, 1] to prevent arccos domain errors from floating-point drift
    cosang = np.clip(np.dot(a, b) / (an * bn), -1.0, 1.0)
    return np.arccos(cosang)

def detect_saccades_arrays(
    t: np.ndarray,
    vel: np.ndarray,
    vel_thresh: float = 0.5,
    min_duration_s: float = 0.010
) -> List[Tuple[int, int]]:
    """
    Detect saccades from velocity and timestamp arrays.

    Parameters
    ----------
    t : np.ndarray
        Timestamp array in seconds.
    vel : np.ndarray
        Velocity array as returned by compute_velocity_arrays.
    vel_thresh : float
        Velocity threshold in normalised units/second.
    min_duration_s : float
        Minimum saccade duration in seconds.

    Returns
    -------
    list of (int, int)
        Inclusive (start_index, end_index) pairs for each saccade.
    """
    above = vel > vel_thresh
    edges = np.diff(above.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends   = np.where(edges == -1)[0] + 1

    if above[0]:
        starts = np.concatenate(([0], starts))
    if above[-1]:
        ends = np.concatenate((ends, [len(vel)]))

    saccades = []
    for s, e in zip(starts, ends):
        if t[e - 1] - t[s] >= min_duration_s:
            saccades.append((s, e - 1))

    return saccades

def analyze_saccades_arrays(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    saccades: List[Tuple[int, int]],
    grid: np.ndarray
) -> List[dict]:
    """
    Extract kinematic metrics for each saccade directly from arrays.

    Parameters
    ----------
    t : np.ndarray
        Timestamp array in seconds.
    x, y : np.ndarray
        Gaze coordinate arrays in normalised units.
    saccades : list of (int, int)
        Saccade index pairs from detect_saccades_arrays.
    grid : np.ndarray
        Shape (15, 2) array of expected target coordinates.

    Returns
    -------
    list of dict
        One dict per saccade with timing, position, and error metrics.
    """
    results = []

    for s_idx, e_idx in saccades:

        t_on  = t[s_idx]
        t_off = t[e_idx]

        # Launch point: sample just before saccade onset
        start_x = x[s_idx - 1] if s_idx > 0 else x[s_idx]
        start_y = y[s_idx - 1] if s_idx > 0 else y[s_idx]
        end_x   = x[e_idx]
        end_y   = y[e_idx]

        # Find nearest grid vertex to the saccade landing point
        dists     = np.sqrt((grid[:, 0] - end_x)**2 + (grid[:, 1] - end_y)**2)
        target_id = int(np.argmin(dists))
        vx, vy    = grid[target_id]

        landing_ed  = np.sqrt((end_x - vx)**2 + (end_y - vy)**2)
        desired_vec = np.array([vx - start_x, vy - start_y])
        actual_vec  = np.array([end_x - start_x, end_y - start_y])
        dir_err_rad = angle_between(desired_vec, actual_vec)
        amplitude   = np.linalg.norm(actual_vec)

        results.append({
            "s_idx":               s_idx,
            "e_idx":               e_idx,
            "onset_time":          t_on,
            "offset_time":         t_off,
            "start":               (start_x, start_y),
            "end":                 (end_x, end_y),
            "target_id":           target_id,
            "landing_error_norm":  landing_ed,
            "direction_error_rad": dir_err_rad,
            "amplitude_norm":      amplitude,
        })

    return results

def compute_centroid_error(
    centroids_world: List[Tuple[float, float]],
    grid: np.ndarray,
    task_name: str,
    max_assign_dist: float = 0.3
) -> List[dict]:
    """
    Match extracted DBSCAN centroids to the expected vertices for a specific
    pursuit path task, and compute spatial error per vertex.

    Only the grid vertices that appear in the task's path are used for
    matching — other vertices are irrelevant since the subject never visited
    them. Each task vertex is assigned its nearest centroid within
    max_assign_dist; unmatched vertices are marked as unvisited.

    Parameters
    ----------
    centroids_world : list of (x, y)
        Centroids returned by extract_centroids_spatiotemporal.
    grid : np.ndarray
        Full array of shape (15, 2) with all grid vertex coordinates.
    task_name : str
        Key into ADPIE.pursuit_path e.g. "Pursuit_path_recollection1".
        Used to look up which vertex indices are relevant for this task.
    max_assign_dist : float
        Maximum normalised distance for a valid centroid-to-vertex assignment.
        Grid columns are ~0.368 apart, so 0.3 is roughly half a column width.

    Returns
    -------
    list of dict
        One dict per task vertex (not all 15) with keys:
            vertex_id       : global index into _grid (0..14)
            path_order      : position of this vertex in the task path (0-based)
            vertex_x/y      : expected target coordinate
            centroid_x/y    : matched centroid coordinate (np.nan if unvisited)
            error_x/y       : signed error (centroid - vertex) (np.nan if unvisited)
            error_dist      : Euclidean distance error in normalised units
            assigned        : bool — whether a centroid was close enough
    """

    # Look up which vertices are relevant for this specific task
    path      = ADPIE.pursuit_path[task_name]
    point_ids = path['point']                          # e.g. [7, 12, 13, 14]

    # Deduplicate while preserving order — some paths revisit the same vertex
    seen          = set()
    unique_ids    = [p for p in point_ids if not (p in seen or seen.add(p))]
    task_vertices = grid[unique_ids]                   # (M, 2) — only task vertices

    # Early exit — no centroids extracted at all
    if len(centroids_world) == 0:
        return [
            {
                "vertex_id":  v_id,
                "path_order": order,
                "vertex_x":   float(grid[v_id, 0]),
                "vertex_y":   float(grid[v_id, 1]),
                "centroid_x": np.nan,
                "centroid_y": np.nan,
                "error_x":    np.nan,
                "error_y":    np.nan,
                "error_dist": np.nan,
                "assigned":   False,
            }
            for order, v_id in enumerate(unique_ids)
        ]

    centroids_arr = np.array(centroids_world)          # (K, 2)

    # Vectorised distance matrix: (M_task_vertices, K_centroids)
    diff        = task_vertices[:, np.newaxis, :] - centroids_arr[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

    # Nearest centroid index and distance per task vertex
    nearest_idx  = np.argmin(dist_matrix, axis=1)
    nearest_dist = dist_matrix[np.arange(len(unique_ids)), nearest_idx]

    results = []

    for order, (v_id, dist) in enumerate(zip(unique_ids, nearest_dist)):

        vx, vy = grid[v_id]

        if dist <= max_assign_dist:
            cx, cy   = centroids_arr[nearest_idx[order]]
            error_x  = float(cx - vx)
            error_y  = float(cy - vy)
            error_d  = float(dist)
            assigned = True
        else:
            cx = cy = error_x = error_y = error_d = np.nan
            assigned = False

        results.append({
            "vertex_id":  v_id,
            "path_order": order,
            "vertex_x":   float(vx),
            "vertex_y":   float(vy),
            "centroid_x": cx,
            "centroid_y": cy,
            "error_x":    error_x,
            "error_y":    error_y,
            "error_dist": error_d,
            "assigned":   assigned,
        })

    return results

def flatten_results(all_results: list) -> pd.DataFrame:
    """
    Flatten the nested all_results list into a single tabular DataFrame
    suitable for CSV export.

    Each row represents one subject × task × vertex combination, with
    summary saccade metrics aggregated to subject-task level.

    Parameters
    ----------
    all_results : list of dict
        Raw results list returned by the main pipeline loop.

    Returns
    -------
    pd.DataFrame
        Flat table with one row per subject-task-vertex.
    """
    rows = []

    for r in all_results:

        subject_id = r["subject_id"]
        cls        = r["class"]
        task       = r["task"]
        n_saccades = r["n_saccades"]
        n_clusters = r["n_clusters"]

        #  Saccade summary (aggregated to subject-task level) 
        saccades = r["saccades"]
        if saccades:
            mean_landing_error  = float(np.mean([s["landing_error_norm"]  for s in saccades]))
            mean_amplitude      = float(np.mean([s["amplitude_norm"]      for s in saccades]))
            mean_direction_err  = float(np.nanmean([s["direction_error_rad"] for s in saccades]))
        else:
            mean_landing_error = mean_amplitude = mean_direction_err = np.nan

        #  Heatmap comparison (one value per group) 
        comparisons = r["comparisons"]
        comparison_flat = {}
        for grp, metrics in comparisons.items():
            comparison_flat[f"peak_dist_{grp}"] = metrics.get("peak_distance", np.nan)
            comparison_flat[f"kl_div_{grp}"]    = metrics.get("kl_divergence", np.nan)

        #  Centroid error (one row per task vertex) 
        centroid_error = r["centroid_error"]

        if centroid_error:
            for ce in centroid_error:
                row = {
                    "filename":         subject_id+"_"+path,
                    "ID":           subject_id,
                    "label":        class_name,
                    "task":         path,
                    "session":      subject_id[-1],
                    "n_saccades":          n_saccades,
                    "n_clusters":          n_clusters,
                    "mean_landing_error":  mean_landing_error,
                    "mean_amplitude":      mean_amplitude,
                    "mean_direction_err":  mean_direction_err,
                    "vertex_id":           ce["vertex_id"],
                    "path_order":          ce["path_order"],
                    "vertex_x":            ce["vertex_x"],
                    "vertex_y":            ce["vertex_y"],
                    "centroid_x":          ce["centroid_x"],
                    "centroid_y":          ce["centroid_y"],
                    "error_x":             ce["error_x"],
                    "error_y":             ce["error_y"],
                    "error_dist":          ce["error_dist"],
                    "assigned":            ce["assigned"],
                    **comparison_flat,
                }
                rows.append(row)

        else:
            # No centroid data — still store subject-task level metrics
            row = {
                "filename":      subject_id+"_"+path,
                "ID":           subject_id,
                "label":        class_name,
                "task":         path,
                "session":      subject_id[-1],
                "n_saccades":          n_saccades,
                "n_clusters":          n_clusters,
                "mean_landing_error":  mean_landing_error,
                "mean_amplitude":      mean_amplitude,
                "mean_direction_err":  mean_direction_err,
                "vertex_id":           np.nan,
                "path_order":          np.nan,
                "vertex_x":            np.nan,
                "vertex_y":            np.nan,
                "centroid_x":          np.nan,
                "centroid_y":          np.nan,
                "error_x":             np.nan,
                "error_y":             np.nan,
                "error_dist":          np.nan,
                "assigned":            False,
                **comparison_flat,
            }
            rows.append(row)

    return pd.DataFrame(rows)

def levenshtein(seq_a: list, seq_b: list) -> int:
    """
    Compute the Levenshtein (edit) distance between two sequences.

    Counts the minimum number of single-element insertions, deletions,
    or substitutions needed to transform seq_a into seq_b. Used here to
    measure how much a subject's observed fixation order diverges from
    the expected path order.

    Parameters
    ----------
    seq_a : list
        Reference sequence (expected path vertex order).
    seq_b : list
        Query sequence (observed centroid visit order).

    Returns
    -------
    int
        Edit distance. 0 = perfect match, higher = more disordered.
    """
    m, n = len(seq_a), len(seq_b)

    # dp[i][j] = edit distance between seq_a[:i] and seq_b[:j]
    dp = np.zeros((m + 1, n + 1), dtype=int)

    # Base cases: transforming to/from empty sequence
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]          # no operation needed
            else:
                dp[i, j] = 1 + min(
                    dp[i - 1, j],      # deletion
                    dp[i, j - 1],      # insertion
                    dp[i - 1, j - 1]   # substitution
                )

    return int(dp[m, n])

def compute_sequence_error(
    centroid_error: List[dict],
    task_name: str
) -> dict:
    """
    Compare the order in which a subject fixated task vertices against the
    expected path order using Levenshtein edit distance.

    Parameters
    ----------
    centroid_error : list of dict
        Output of compute_centroid_error — one dict per task vertex,
        sorted by path_order, with an 'assigned' flag and 'vertex_id'.
    task_name : str
        Key into ADPIE.pursuit_path e.g. "Pursuit_path_recollection1".

    Returns
    -------
    dict
        "expected_order"    : list of vertex ids in correct path order
        "observed_order"    : list of vertex ids in the order they were visited
        "sequence_error"    : Levenshtein distance between the two
        "n_expected"        : number of unique vertices in the expected path
        "n_observed"        : number of assigned centroids
        "normalised_error"  : sequence_error / n_expected (0=perfect, 1=worst)
    """
    path          = ADPIE.pursuit_path[task_name]
    point_ids     = path['point']

    # Deduplicate expected order while preserving sequence
    seen          = set()
    expected_order = [p for p in point_ids if not (p in seen or seen.add(p))]

    # Observed order: only vertices that were successfully matched to a centroid,
    # sorted by path_order (which reflects the order centroids appear in the path)
    observed_order = [
        c['vertex_id']
        for c in sorted(centroid_error, key=lambda c: c['path_order'])
        if c['assigned']
    ]

    distance = levenshtein(expected_order, observed_order)

    return {
        "expected_order":   expected_order,
        "observed_order":   observed_order,
        "sequence_error":   distance,
        "n_expected":       len(expected_order),
        "n_observed":       len(observed_order),
        "normalised_error": distance / max(len(expected_order), 1),
    }

def compute_gaze_pen_lag(
    centroid_error: List[dict],
    subj_data: dict,
    task_name: str,
    grid: np.ndarray,
    max_assign_dist: float = 0.3
) -> List[dict]:
    """
    Compute the temporal lag between when the eye arrives at each target
    vertex and when the pen arrives at the same vertex.

    A positive lag means the eye arrived before the pen (normal).
    A negative lag means the pen arrived before the eye (reversed coordination).
    NaN means the vertex could not be matched in one or both modalities.

    Gaze arrival time  : onset timestamp of the centroid assigned to the vertex
    Pen arrival time   : timestamp of the pen sample closest to the vertex

    Parameters
    ----------
    centroid_error : list of dict
        Output of compute_centroid_error — contains vertex assignments and
        centroid positions, but NOT onset times. Onset time is recovered
        here from the raw gaze arrays.
    subj_data : dict
        Raw sample dict containing T, X, Y (pen) and gazeX, gazeY, T (gaze).
    task_name : str
        Used to look up expected path vertices.
    grid : np.ndarray
        Full (15, 2) grid coordinate array.
    max_assign_dist : float
        Maximum distance for a pen sample to be considered "at" a vertex.

    Returns
    -------
    list of dict
        One dict per assigned vertex with keys:
            vertex_id           : global grid index
            path_order          : position in task path
            gaze_arrival_time   : timestamp when eye centroid formed (s)
            pen_arrival_time    : timestamp when pen reached vertex (s)
            lag_s               : gaze_arrival_time - pen_arrival_time (s)
                                  positive = eye leads pen (normal)
                                  negative = pen leads eye (atypical)
    """

    #  Extract pen arrays 
    T_pen = np.asarray(subj_data.get("T",  []), dtype=float)
    X_pen = np.asarray(subj_data.get("X",  []), dtype=float)
    Y_pen = np.asarray(subj_data.get("Y",  []), dtype=float)

    #  Extract gaze arrays ─
    T_gaze = np.asarray(subj_data.get("T",     []), dtype=float)
    gaze_x = np.asarray(subj_data.get("gazeX", []), dtype=float)
    gaze_y = np.asarray(subj_data.get("gazeY", []), dtype=float)

    n_pen  = min(len(T_pen),  len(X_pen),  len(Y_pen))
    n_gaze = min(len(T_gaze), len(gaze_x), len(gaze_y))

    T_pen,  X_pen,  Y_pen  = T_pen[:n_pen],   X_pen[:n_pen],   Y_pen[:n_pen]
    T_gaze, gaze_x, gaze_y = T_gaze[:n_gaze], gaze_x[:n_gaze], gaze_y[:n_gaze]

    results = []

    for ce in centroid_error:

        if not ce['assigned']:
            continue

        v_id      = ce['vertex_id']
        vx, vy    = grid[v_id]

        #  Gaze arrival time
        # Find the first gaze sample inside this centroid's spatial region.
        # The centroid centre is (centroid_x, centroid_y); use a radius equal
        # to max_assign_dist to define "arrived at vertex".
        gaze_dists   = np.sqrt((gaze_x - vx)**2 + (gaze_y - vy)**2)
        gaze_near    = np.where(gaze_dists <= max_assign_dist)[0]

        if len(gaze_near) == 0:
            gaze_arrival = np.nan
        else:
            # First sample where gaze entered the vertex region
            gaze_arrival = float(T_gaze[gaze_near[0]])

        #  Pen arrival time 
        # Find the pen sample that passed closest to this vertex
        pen_dists  = np.sqrt((X_pen - vx)**2 + (Y_pen - vy)**2)
        nearest_pen = int(np.argmin(pen_dists))

        if pen_dists[nearest_pen] > max_assign_dist:
            pen_arrival = np.nan
        else:
            pen_arrival = float(T_pen[nearest_pen])

        #  Lag ─
        if np.isnan(gaze_arrival) or np.isnan(pen_arrival):
            lag = np.nan
        else:
            lag = gaze_arrival - pen_arrival

        results.append({
            "vertex_id":         v_id,
            "path_order":        ce['path_order'],
            "vertex_x":          vx,
            "vertex_y":          vy,
            "gaze_arrival_time": gaze_arrival,
            "pen_arrival_time":  pen_arrival,
            "lag_s":             lag,
        })

    return results

def summarise_lag(lag_results: List[dict]) -> dict:
    """
    Aggregate per-vertex lag values into subject-level summary statistics.

    Parameters
    ----------
    lag_results : list of dict
        Output of compute_gaze_pen_lag.

    Returns
    -------
    dict
        mean_lag_s      : average lag across vertices (positive = eye leads)
        std_lag_s       : variability — high std suggests erratic coordination
        n_valid         : number of vertices with valid (non-NaN) lag
        n_eye_leads     : vertices where eye arrived before pen (normal)
        n_pen_leads     : vertices where pen arrived before eye (atypical)
        reversed_ratio  : proportion of vertices where pen led the eye
    """
    lags = np.array([r["lag_s"] for r in lag_results], dtype=float)
    valid = lags[~np.isnan(lags)]

    if len(valid) == 0:
        return {
            "mean_lag_s":     np.nan,
            "std_lag_s":      np.nan,
            "n_valid":        0,
            "n_eye_leads":    0,
            "n_pen_leads":    0,
            "reversed_ratio": np.nan,
        }

    return {
        "mean_lag_s":     float(np.mean(valid)),
        "std_lag_s":      float(np.std(valid)),
        "n_valid":        int(len(valid)),
        "n_eye_leads":    int(np.sum(valid > 0)),
        "n_pen_leads":    int(np.sum(valid < 0)),
        "reversed_ratio": float(np.sum(valid < 0) / len(valid)),
    }

def main_sequence_lag():
    """
    Pipeline to compute sequence error and gaze-pen lag features
    for each subject and task, saved to a CSV.

    Pipeline stages
    ---------------
    1. For each task and class, load subject data
    2. Extract spatiotemporal centroids via DBSCAN
    3. Match centroids to expected task vertices
    4. Compute fixation sequence error (Levenshtein)
    5. Compute gaze-pen lag per vertex
    6. Save all results to CSV
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

    class_names   = ["MCI_AD", "PD", "CTL"]
    num_vertices  = [len(list(set(gt['point']))) for gt in ADPIE.pursuit_path.values()]

    rows = []

    for path, expected_clusters in zip(gt_base_names, num_vertices):
        print(f"\nProcessing: {path}  (expected vertices: {expected_clusters})")

        ad1  = pm.get_single_class(path, "MCI_AD", subject_id=True)
        pdn1 = pm.get_single_class(path, "PD",     subject_id=True)
        ctl1 = pm.get_single_class(path, "CTL",    subject_id=True)

        for class_name, nd in zip(class_names, [ad1, pdn1, ctl1]):
            print(f"  [{class_name}] {len(nd)} subjects")

            for data in nd:

                subject_id = data.get("subject_id") or data.get("id", "unknown")

                #  Base row shared across all result types 
                base = {
                    "filename":         subject_id+"_"+path,
                    "ID":           subject_id,
                    "label":        class_name,
                    "task":         path,
                    "session":      subject_id[-1],
                    "expected_clusters":  expected_clusters,
                }

                #  Centroid extraction 
                try:
                    centroids, labels, k = extract_centroids_spatiotemporal(data)
                    noise_ratio = float(np.sum(labels == -1) / len(labels))
                except Exception as e:
                    print(f"    Centroid extraction failed — {subject_id}: {e}")
                    rows.append({
                        **base,
                        "n_clusters":         0,
                        "noise_ratio":        np.nan,
                        "expected_order":     None,
                        "observed_order":     None,
                        "sequence_error":     np.nan,
                        "normalised_seq_err": np.nan,
                        "n_expected":         np.nan,
                        "n_observed":         np.nan,
                        "mean_lag_s":         np.nan,
                        "std_lag_s":          np.nan,
                        "n_valid_lag":        np.nan,
                        "n_eye_leads":        np.nan,
                        "n_pen_leads":        np.nan,
                        "reversed_ratio":     np.nan,
                        "error":              str(e),
                    })
                    continue

                #  Centroid-to-vertex matching 
                try:
                    centroid_error = compute_centroid_error(
                        centroids, _grid, task_name=path
                    )
                except Exception as e:
                    print(f"    Centroid matching failed — {subject_id}: {e}")
                    rows.append({
                        **base,
                        "n_clusters":         k,
                        "noise_ratio":        noise_ratio,
                        "expected_order":     None,
                        "observed_order":     None,
                        "sequence_error":     np.nan,
                        "normalised_seq_err": np.nan,
                        "n_expected":         np.nan,
                        "n_observed":         np.nan,
                        "mean_lag_s":         np.nan,
                        "std_lag_s":          np.nan,
                        "n_valid_lag":        np.nan,
                        "n_eye_leads":        np.nan,
                        "n_pen_leads":        np.nan,
                        "reversed_ratio":     np.nan,
                        "error":              str(e),
                    })
                    continue

                #  Sequence error ─
                seq = {}
                try:
                    seq = compute_sequence_error(centroid_error, path)
                    print(
                        f"    {subject_id} | "
                        f"seq_err={seq['sequence_error']} "
                        f"({seq['normalised_error']:.2f}) | "
                        f"expected={seq['expected_order']} "
                        f"observed={seq['observed_order']}"
                    )
                except Exception as e:
                    print(f"    Sequence error failed — {subject_id}: {e}")

                #  Gaze–pen lag ─
                lag_summary = {}
                try:
                    lag_results = compute_gaze_pen_lag(
                        centroid_error, data, path, _grid
                    )
                    lag_summary = summarise_lag(lag_results)
                    print(
                        f"    {subject_id} | "
                        f"mean_lag={lag_summary['mean_lag_s']:.3f}s "
                        f"std={lag_summary['std_lag_s']:.3f}s | "
                        f"eye_leads={lag_summary['n_eye_leads']}/"
                        f"{lag_summary['n_valid']}"
                    )
                except Exception as e:
                    print(f"    Gaze–pen lag failed — {subject_id}: {e}")

                #  Collect row 
                rows.append({
                    **base,
                    "n_clusters":         k,
                    "noise_ratio":        round(noise_ratio, 4),
                    "expected_order":     str(seq.get("expected_order", [])),
                    "observed_order":     str(seq.get("observed_order", [])),
                    "sequence_error":     seq.get("sequence_error",    np.nan),
                    "normalised_seq_err": round(seq.get("normalised_error", np.nan), 4),
                    "n_expected":         seq.get("n_expected",        np.nan),
                    "n_observed":         seq.get("n_observed",        np.nan),
                    "mean_lag_s":         round(lag_summary.get("mean_lag_s",     np.nan), 4),
                    "std_lag_s":          round(lag_summary.get("std_lag_s",      np.nan), 4),
                    "n_valid_lag":        lag_summary.get("n_valid",      np.nan),
                    "n_eye_leads":        lag_summary.get("n_eye_leads",  np.nan),
                    "n_pen_leads":        lag_summary.get("n_pen_leads",  np.nan),
                    "reversed_ratio":     round(lag_summary.get("reversed_ratio", np.nan), 4),
                    "error":              None,
                })

    #  Write CSV ─

    out_path   = "sequence_lag_features.csv"
    fieldnames = [
        "filename", "ID", "label", "task", "session", 
        "n_clusters", "noise_ratio", "expected_clusters",
        "expected_order", "observed_order",
        "sequence_error", "normalised_seq_err", "n_expected", "n_observed",
        "mean_lag_s", "std_lag_s", "n_valid_lag",
        "n_eye_leads", "n_pen_leads", "reversed_ratio",
        "error",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows → {os.path.abspath(out_path)}")
    return rows

