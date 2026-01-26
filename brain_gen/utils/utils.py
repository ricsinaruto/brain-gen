import numpy as np
from scipy.spatial import distance_matrix


def key_args_str(_key_args: dict[str, str]) -> str:
    """Convert key args to a single string."""
    key_args = [f"{k}-{v}" for k, v in _key_args.items()]
    key_args = [arg.replace("_", "") for arg in key_args]
    return "_".join(key_args)


def compute_roi_layout_2d(labels, src):
    """Compute a 2D layout for ROI indices based on their 3D centroids.

    Parameters ---------- labels : list of mne.Label     Parcel labels used for
    extract_label_time_course. The order of this     list defines the ROI index
    (0..n_labels-1). src : mne.SourceSpaces     The source space used to create the
    forward model (fsaverage surface).     Assumed to be a standard 2-hemisphere surface
    source space.

    Returns ------- layout : np.ndarray, shape (n_labels, 2)     Integer (y, x)
    coordinates for each ROI index. So for ROI i,     layout[i] = (y_i, x_i) is the
    pixel where you can place its value.
    """
    # Map hemisphere string to src index
    hemi_to_idx = {"lh": 0, "rh": 1}

    # For faster lookup: surface vertex indices -> row indices in src['rr']
    vertno_maps = {}
    for hemi_str, hemi_idx in hemi_to_idx.items():
        vertno = src[hemi_idx]["vertno"]  # vertex numbers on the surface
        # map surface vertex number -> index into src[hemi]["rr"]
        vertno_maps[hemi_str] = {v: i for i, v in enumerate(vertno)}

    centroids = []

    for lab in labels:
        hemi = lab.hemi
        if hemi not in hemi_to_idx:
            raise ValueError(f"Unexpected label hemisphere: {hemi!r}")

        hemi_idx = hemi_to_idx[hemi]
        rr = src[hemi_idx]["rr"]  # (n_verts_hemi, 3)
        vmap = vertno_maps[hemi]

        # Intersect label vertices with source space vertices
        rr_indices = []
        for v in lab.vertices:
            idx = vmap.get(v, None)
            if idx is not None:
                rr_indices.append(idx)

        if len(rr_indices) == 0:
            # Fallback: just use the mean of all src vertices for that hemi
            # (should almost never happen if labels match the annot/src)
            centroid = rr.mean(axis=0)
        else:
            centroid = rr[np.array(rr_indices)].mean(axis=0)

        centroids.append(centroid)

    centroids = np.asarray(centroids, dtype=float)  # (n_labels, 3)

    # --- 3D -> 2D via PCA (SVD) ---
    X = centroids - centroids.mean(axis=0, keepdims=True)
    # SVD: X = U S V^T, principal directions = rows of V^T
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    # Project onto first two principal directions
    coords_2d = X @ Vt[:2].T  # (n_labels, 2)

    # Normalize to [0, 1] in both dimensions
    mins = coords_2d.min(axis=0, keepdims=True)
    maxs = coords_2d.max(axis=0, keepdims=True)
    rng = np.maximum(maxs - mins, 1e-9)  # avoid division by zero
    coords_norm = (coords_2d - mins) / rng

    D3 = distance_matrix(centroids, centroids)
    D2 = distance_matrix(coords_2d, coords_2d)
    corr = np.corrcoef(D3.ravel(), D2.ravel())[0, 1]
    print(f"INFO: Correlation between 3D and 2D distances: {corr}")

    # Map to integer pixel coordinates
    # xs = np.round(coords_norm[:, 0] * (W - 1)).astype(int)
    # ys = np.round(coords_norm[:, 1] * (H - 1)).astype(int)

    return coords_norm
