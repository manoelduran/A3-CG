from dataclasses import dataclass


@dataclass
class SegmentParams:
    min_area: int = 10000
    max_area: int = 2_000_000
    open_ksize: int = 7
    sure_bg_dilate: int = 3
    distance_thresh: float = 0.3
    n_clusters: int = 3  # Number of clusters for K-means (background + bean colors)
    cluster_selection_method: str = "size"  # "size" or "brightness" - how to select bean clusters
