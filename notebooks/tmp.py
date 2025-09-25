import numpy as np
from scipy.ndimage import gaussian_filter

# --- Helpers ---
def sanitize_pd(pd_array, cap=None):
    """
    Convert to numpy array (n x 2). If cap is provided, replace np.inf with cap.
    """
    arr = np.array(pd_array, dtype=float)
    if arr.size == 0:
        return arr.reshape((0,2))
    if cap is not None:
        arr[np.isinf(arr[:,1]),1] = cap
    return arr

# Triangular (tent) function for an interval (b,d) evaluated at t (array)
def triangular(b, d, t):
    mid = (b + d) / 2.0
    height = max(0.0, (d - b) / 2.0)
    out = np.zeros_like(t, dtype=float)
    left_mask = (t >= b) & (t <= mid)
    right_mask = (t > mid) & (t <= d)
    if height > 0:
        out[left_mask] = (t[left_mask] - b)
        out[right_mask] = (d - t[right_mask])
    return out.clip(min=0.0)

# --- Vectorizations ---
def betti_curve(pd_pairs, thresholds, cap=None):
    """
    Compute Betti curve from PD: counts of intervals alive at each threshold.
    """
    pd_arr = sanitize_pd(pd_pairs, cap=cap)
    t = np.asarray(thresholds)
    counts = np.zeros_like(t, dtype=int)
    for (b,d) in pd_arr:
        counts += ((t >= b) & (t < d)).astype(int)
    return counts

def persistence_landscape(pd_pairs, thresholds, num_layers=3, cap=None):
    """
    Compute persistence landscape: for each threshold produce the vector of k-th largest triangular values.
    Returns array shape (num_layers, len(thresholds)).
    """
    pd_arr = sanitize_pd(pd_pairs, cap=cap)
    t = np.asarray(thresholds)
    if pd_arr.shape[0] == 0:
        return np.zeros((num_layers, len(t)))
    mats = []
    for (b,d) in pd_arr:
        mats.append(triangular(b, d, t))
    mats = np.vstack(mats)
    sorted_vals = -np.sort(-mats, axis=0)  # descending
    n_int = mats.shape[0]
    if n_int < num_layers:
        pad = np.zeros((num_layers - n_int, mats.shape[1]))
        sorted_vals = np.vstack([sorted_vals, pad])
    landscape = np.zeros((num_layers, mats.shape[1]))
    for k in range(num_layers):
        landscape[k] = sorted_vals[k]
    return landscape

def silhouette(pd_pairs, thresholds, p=1.0, cap=None):
    """
    Silhouette function: weighted average of triangular functions.
    weights = persistence^p
    """
    pd_arr = sanitize_pd(pd_pairs, cap=cap)
    t = np.asarray(thresholds)
    if pd_arr.shape[0] == 0:
        return np.zeros_like(t, dtype=float)
    weights = np.maximum(0.0, (pd_arr[:,1] - pd_arr[:,0])) ** p
    numer = np.zeros_like(t, dtype=float)
    denom = np.sum(weights)
    for (w,(b,d)) in zip(weights, pd_arr):
        numer += w * triangular(b, d, t)
    if denom == 0:
        return np.zeros_like(t, dtype=float)
    return numer / denom

def persistence_image(pd_pairs, birth_range=None, pers_range=None, 
                      birth_bins=50, pers_bins=50, sigma=1.0, cap=None):
    """
    Persistence Image: map (birth, persistence=d-b) to a 2D grid and smooth with Gaussian kernel.
    Returns:
      img: 2D numpy array (pers_bins x birth_bins)
      extent: (birth_min, birth_max, pers_min, pers_max) for imshow if needed
    """
    pd_arr = sanitize_pd(pd_pairs, cap=cap)
    if pd_arr.shape[0] == 0:
        if birth_range is None: birth_range = (0,1)
        if pers_range is None: pers_range = (0,1)
        return np.zeros((pers_bins, birth_bins)), (birth_range[0], birth_range[1], pers_range[0], pers_range[1])
    births = pd_arr[:,0]
    pers = pd_arr[:,1] - pd_arr[:,0]
    mask = pers > 0
    births, pers = births[mask], pers[mask]
    if birth_range is None:
        bmin, bmax = births.min(), births.max()
        bpad = 0.05 * (bmax - bmin + 1e-8)
        birth_range = (bmin - bpad, bmax + bpad)
    if pers_range is None:
        pmin, pmax = pers.min(), pers.max()
        ppad = 0.05 * (pmax - pmin + 1e-8)
        pers_range = (max(0.0, pmin - ppad), pmax + ppad)
    birth_edges = np.linspace(birth_range[0], birth_range[1], birth_bins+1)
    pers_edges = np.linspace(pers_range[0], pers_range[1], pers_bins+1)
    H, _, _ = np.histogram2d(pers, births, bins=[pers_edges, birth_edges], weights=pers)
    img = gaussian_filter(H, sigma=sigma)
    extent = (birth_range[0], birth_range[1], pers_range[0], pers_range[1])
    return img, extent



# Test with example PD
pd = [(0.0, 0.6), (0.2, 1.0), (0.5, np.inf)]
t = np.linspace(0, 1.5, 151)

betti = betti_curve(pd, t, cap=1.5)
land = persistence_landscape(pd, t, num_layers=3, cap=1.5)
sil = silhouette(pd, t, p=1.0, cap=1.5)
img, extent = persistence_image(pd, birth_bins=80, pers_bins=60, sigma=1.0, cap=1.5)
