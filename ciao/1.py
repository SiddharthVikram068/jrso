"""
Quick loader for the Ciao dataset (Tang et al.).
The .mat files are MATLAB v5, so we use scipy.io.loadmat (not h5py).

trustnetwork.mat -> 'trustnetwork' : (111781, 2) array of [user_id, trustee_id]
rating.mat       -> 'rating'       : (284086, 5) array of
                                      [userId, productId, categoryId,
                                       rating(0-5), helpfulness]
"""

from pathlib import Path
import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).parent

trust = loadmat(HERE / "trustnetwork.mat")
rating = loadmat(HERE / "rating.mat")

print("trustnetwork.mat keys :", [k for k in trust if not k.startswith("__")])
print("rating.mat keys       :", [k for k in rating if not k.startswith("__")])

trust_edges = trust["trustnetwork"]
ratings = rating["rating"]

print("\n--- trust network ---")
print("shape :", trust_edges.shape, "dtype:", trust_edges.dtype)
print("sample:\n", trust_edges[:5])
print("#users in trust graph:", len(np.unique(trust_edges)))

print("\n--- ratings ---")
print("shape :", ratings.shape, "dtype:", ratings.dtype)
print("sample:\n", ratings[:5])
print("#users :", len(np.unique(ratings[:, 0])))
print("#items :", len(np.unique(ratings[:, 1])))
print("rating range:", ratings[:, 3].min(), "to", ratings[:, 3].max())
