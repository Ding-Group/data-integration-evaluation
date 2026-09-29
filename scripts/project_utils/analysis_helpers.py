import matplotlib.pyplot as plt
import colorcet as cc
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from pathlib import Path

from adata_utils import EMB_KEY


NAME_MAP = {
    'cca': "CCA", 'clear': "CLEAR", 'cyclone': "CYCLONE", 'equidistant': 'Equidistant',
    'gedi': "GEDI", 'geneformer': 'GeneFormer', 'geneptw': 'GenePT-W', 'harmony': 'Harmony',
    'imap': "iMAP", 'insct': 'INSCT',  'islander': 'Islander', 'langcell': "LangCell",
    'liger': "LIGER", 'mnn': "MNN", 'ocat': "OCAT", 'oneBigBlob': 'oneBigBlob',
    'pca': "PCA", 'portal': "Portal", 'respan': "ResPAN", 'rpca': "RPCA",
    'saucie': "SAUCIE", 'scalex': "SCALEX", 'scalign': "scAlign", 'scanorama': "Scanorama", 
    'scanvi': "scANVI", 'sccobra': "scCobra", 'sccraft': "scCRAFT", 'scdml': 'scDML',
    'scdreamer': 'scDREAMER', 'scetm': 'scETM', 'scgpt': 'scGPT', 'scmerge': 'scMerge',
    'scmulan': "scMulan", 'scphere': 'scPhere', 'scprint': 'scPRINT', 'scvi': "scVI", 
    'simba': "SIMBA", 'sysvi': "sysVI", 'uce': "UCE"
}

integrations = sorted(NAME_MAP.values(), key=lambda k: k.lower())
palette = cc.glasbey[:len(integrations)]
INTEGRATION_COLOR_DICT = {key: palette[i] for i, key in enumerate(integrations)}


def find_dir(h, base='/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/work/'):
    a,b = h.split('/')
    base = Path(f"{base}/{a}/")
    prefix = Path(b)
    dirs = [
        p for p in base.glob("*")
        if p.is_dir() and p.relative_to(base).as_posix().startswith(prefix.as_posix())
    ]
    return dirs[0]


def plot_silhouette(adata_int, celltype_col):
    cluster_labels = adata_int.obs[celltype_col].cat.codes
    X = adata_int.obsm[EMB_KEY]
    n_clusters = len(np.unique(cluster_labels))

    # Create a subplot with 1 row and 2 columns
    fig, ax = plt.subplots(1, 1, figsize=(6, 9))

    # 1. Silhouette Plot Setup
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # # Calculate average score and individual sample scores
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    cmap = adata_int.uns['CellType_colors']

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate and sort silhouette scores for cluster i
        ith_cluster_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_values.sort()

        size_cluster_i = ith_cluster_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cmap[i]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

        y_lower = y_upper + 10

    ax.set_title("Silhouette Plot")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])

    plt.tight_layout()
    plt.show()