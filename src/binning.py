"""Phase-aware standardisation bins (used by stages 05 and 05b).
assign_bins() cuts the pressure percentile into bins separately inside every (competition, innings/phase) group, then merges any bin with fewer than
`min_n` balls into its lower neighbour (the top bin into the one below it), so that every standardisation cell stays well populated."""
import numpy as np

def assign_bins(df, groupcols, x, edges, min_n=400):
    x = np.asarray(x, float); out = np.zeros(len(df), dtype=int)
    for key, idx in df.reset_index(drop=True).groupby(groupcols).indices.items():
        e = list(edges); xi = x[idx]
        while True:
            b = np.digitize(xi, e[1:-1], right=True); cnt = np.bincount(b, minlength=len(e) - 1)
            if len(e) <= 2 or cnt.min() >= min_n: break
            j = int(np.argmin(cnt)); e.pop(j if j > 0 else 1)              # merge the smallest bin with its lower neighbour (bin 0 with its upper one)
        out[idx] = np.digitize(xi, e[1:-1], right=True)
    return out
