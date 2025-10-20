from dataclasses import dataclass


@dataclass
class SegmentParams:
    min_area: int = 600
    max_area: int = 2_000_000
    open_ksize: int = 5
    sure_bg_dilate: int = 5
    distance_thresh: float = 0.25
