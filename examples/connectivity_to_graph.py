"""
===============================================
Export a Connectivity object to a NetworkX graph
===============================================

Converts a `mne_connectivity.Connectivity` object into a `networkx.Graph` so that the whole `NetworkX <https://networkx.org>`__ ecosystem (graph-theoretical
metrics, community detection, layout algorithms, graph file formats, ...) becomes available.

We compute sensor-space spectral connectivity in the alpha band on the MNE `sample dataset`, export it to NetworkX and run a few classical graph analyses on it.
"""

# Authors: Raphaël Bordas <bordasraph@gmail.com>
#          Marie-Constance Corsi <marie-constance.corsi@inria.fr>
#
# License: BSD (3-clause)

# %%

import os.path as op

import matplotlib.pyplot as plt
import mne
import networkx as nx
import numpy as np
from mne.datasets import sample

from examples.connectivity_classes import sfreq
from mne_connectivity import spectral_connectivity_epochs

print(__doc__)

# %%
# Load the data and build epochs
# ------------------------------
#
data_path = sample.data_path()
raw_fname = op.join(data_path, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif")
event_fname = op.join(data_path, "MEG", "sample", "sample_audvis_filt-0-40_raw-eve.fif")

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

# %%
# Add a bad channel
raw.info["bads"] += ["MEG 2443"]

# To keep the example fast (and the graph figures readable) we only keep a handful of gradiometers spread over the helmet. In a real analysis you would of course keep all sensors or, even better, work in source space with anatomical labels as nodes.
picks = mne.pick_types(raw.info, meg="grad", eeg=False, stim=False, exclude="bads")
picks = picks[::16]  # ~13 sensors, evenly spread over the array
raw.pick(picks)


# Create epochs for the auditory condition
event_id, tmin, tmax = {"Auditory/Left": 1}, -0.2, 0.5
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    baseline=(None, 0),
    reject=dict(grad=4000e-13),
    preload=True,
)
print(epochs)
# TODO: epochs.load_data().pick("grad")  # just keep MEG and no EOG now

# %%
# Compute connectivity
# --------------------
#
# We estimate the (debiased) weighted phase lag index in the alpha band, on the post-stimulus window only. ``faverage=True`` averages over the band so that the resulting :class:`~mne_connectivity.Connectivity` object holds a single value per channel pair -- i.e. exactly one edge weight per pair.
fmin, fmax = 8.0, 13.0
sfreq = epochs.info["sfreq"]

con = spectral_connectivity_epochs(
    epochs,
    method="wpli2_debiased",
    mode="multitaper",
    sfreq=sfreq,
    fmin=fmin,
    fmax=fmax,
    faverage=True,
    tmin=0.0,
    mt_adaptive=False,
    n_jobs=1,
)
print(con)

# %%
# Convert to a NetworkX graph
# ---------------------------
#
# `mne_connectivity.to_networkx` returns an undirected `networkx.Graph` for symmetric measures (coherence, PLV, wPLI, ...) and a `networkx.DiGraph` for directed ones (Granger causality, PDC, ...).
# The node names are taken from `con.names` and the connectivity values are stored as the "weight" edge attribute.
list_graph = con.to_networkx(is_directed=False, is_weighted=True)
graph=list_graph[0]
print(f"Graph type       : {type(graph).__name__}")
print(f"Number of nodes  : {graph.number_of_nodes()}")
print(f"Number of edges  : {graph.number_of_edges()}")
print(f"Directed         : {graph.is_directed()}")


# %%
# Nodes are labelled with the channel names, edges carry the connectivity value:
print("First 5 nodes:", list(graph.nodes)[:5])
print("First 5 edges:")
for u, v, w in list(graph.edges(data="weight"))[:5]:
    print(f"  {u} -- {v}: {w:.3f}")

# %%
# Thresholding the graph
# ----------------------
#
# All-to-all connectivity gives a fully connected graph, on which most graph-theoretical metrics are not very informative. A possible approach is to keep only the strongest edges (here the top 20 %). NetworkX makes this a one-liner.
weights = np.array([w for _, _, w in graph.edges(data="weight")])
threshold = np.percentile(weights, 80)

strong = nx.Graph(
    ((u, v, d) for u, v, d in graph.edges(data=True) if d["weight"] >= threshold)
)
strong.add_nodes_from(graph.nodes(data=True))  # keep isolated nodes
print(f"Kept {strong.number_of_edges()} / {graph.number_of_edges()} edges "
      f"(threshold = {threshold:.3f})")


# %%
# Graph-theoretical metrics
# -------------------------
#
# Once the object is a NetworkX graph, any graph measure is directly available. Here we compute the node strength (weighted degree), the weighted clustering coefficient and the betweenness centrality.
strength = dict(graph.degree(weight="weight"))
clustering = nx.clustering(graph, weight="weight")
betweenness = nx.betweenness_centrality(strong, weight="weight")

print(f"{'channel':<12}{'strength':>10}{'clustering':>12}{'betweenness':>13}")
for ch in graph.nodes:
    print(f"{ch:<12}{strength[ch]:>10.2f}{clustering[ch]:>12.3f}"
          f"{betweenness[ch]:>13.3f}")

# %%
# Community detection (here with the Louvain algorithm) partitions the sensors
# into groups that are more strongly connected among themselves than with the
# rest of the array.
communities = nx.community.louvain_communities(strong, weight="weight", seed=42)
for i, comm in enumerate(communities):
    print(f"Community {i}: {sorted(comm)}")

# %%
# Plot the graph on the sensor layout
# -----------------------------------
#
# Because the nodes are named after the channels, we can use the actual sensor positions as the NetworkX layout, which makes the graph directly interpretable in terms of head topography.
layout = mne.channels.find_layout(epochs.info)
pos = {
    name: layout.pos[layout.names.index(name)][:2]
    for name in graph.nodes
    if name in layout.names
}

node_color = np.zeros(len(graph))
for i, comm in enumerate(communities):
    for node in comm:
        node_color[list(graph.nodes).index(node)] = i

fig, ax = plt.subplots(figsize=(7, 7))
edge_weights = np.array([d["weight"] for _, _, d in strong.edges(data=True)])
nx.draw_networkx_edges(
    strong,
    pos,
    ax=ax,
    width=3 * edge_weights / edge_weights.max(),
    alpha=0.6,
    edge_color=edge_weights,
    edge_cmap=plt.cm.viridis,
)
nx.draw_networkx_nodes(
    strong,
    pos,
    ax=ax,
    node_size=[3000 * strength[n] / max(strength.values()) for n in strong.nodes],
    node_color=node_color,
    cmap=plt.cm.Set2,
)
nx.draw_networkx_labels(strong, pos, ax=ax, font_size=7)
ax.set_title("Alpha-band (dPLI) network, top 20% of edges\n"
             "node size = strength, colour = Louvain community")
ax.set_axis_off()
fig.tight_layout()

# %%
# Saving the graph
# ----------------

df = nx.to_pandas_edgelist(graph)
print(df.head())
