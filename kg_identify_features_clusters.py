import re
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform

# --- Step 1: Parse similarity lines ---
lines = """
11.2.1 <-> 11.2.2 = 0.494
11.2.1 <-> 11.2.3 = 0.772
11.2.1 <-> 11.2.4 = 0.761
11.2.1 <-> 11.2.5 = 0.601
11.2.1 <-> 11.2.6 = 0.705
11.2.1 <-> 11.2.7 = 0.785
11.2.1 <-> 11.2.8 = 0.660
11.2.1 <-> 11.2.9 = 0.658
11.2.2 <-> 11.2.3 = 0.639
11.2.2 <-> 11.2.4 = 0.530
11.2.2 <-> 11.2.5 = 0.464
11.2.2 <-> 11.2.6 = 0.482
11.2.2 <-> 11.2.7 = 0.597
11.2.2 <-> 11.2.8 = 0.583
11.2.2 <-> 11.2.9 = 0.507
11.2.3 <-> 11.2.4 = 0.818
11.2.3 <-> 11.2.5 = 0.588
11.2.3 <-> 11.2.6 = 0.759
11.2.3 <-> 11.2.7 = 0.902
11.2.3 <-> 11.2.8 = 0.843
11.2.3 <-> 11.2.9 = 0.754
11.2.4 <-> 11.2.5 = 0.760
11.2.4 <-> 11.2.6 = 0.694
11.2.4 <-> 11.2.7 = 0.863
11.2.4 <-> 11.2.8 = 0.830
11.2.4 <-> 11.2.9 = 0.662
11.2.5 <-> 11.2.6 = 0.506
11.2.5 <-> 11.2.7 = 0.696
11.2.5 <-> 11.2.8 = 0.541
11.2.5 <-> 11.2.9 = 0.504
11.2.6 <-> 11.2.7 = 0.722
11.2.6 <-> 11.2.8 = 0.671
11.2.6 <-> 11.2.9 = 0.644
11.2.7 <-> 11.2.8 = 0.799
11.2.7 <-> 11.2.9 = 0.693
11.2.8 <-> 11.2.9 = 0.673

""".strip().splitlines()

pattern = re.compile(r"([\d\.]+)\s+<->\s+([\d\.]+)\s+=\s+([0-9.]+)")
pairs = [pattern.findall(line)[0] for line in lines]

# --- Step 2: Build list of sections ---
sections = sorted({s for p in pairs for s in p[:2]})
n = len(sections)
section_index = {s: i for i, s in enumerate(sections)}

# --- Step 3: Build full similarity matrix ---
sim_matrix = np.eye(n)
for a, b, s in pairs:
    i, j = section_index[a], section_index[b]
    sim_matrix[i, j] = sim_matrix[j, i] = float(s)

# Convert to distance matrix (1 - similarity)
dist_matrix = 1 - sim_matrix

# --- Step 4: Agglomerative clustering (corrected API) ---
clustering = AgglomerativeClustering(
    n_clusters=None,
    metric="precomputed",     # new parameter name
    linkage="average",
    distance_threshold=0.20
)
labels = clustering.fit_predict(dist_matrix)

# --- Step 5: Display clusters ---
clusters = pd.DataFrame({"Section": sections, "Cluster": labels}).sort_values("Cluster")
print(clusters)

