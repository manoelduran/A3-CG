import cv2
import os
import numpy as np
from cv2.typing import MatLike
from sklearn.cluster import KMeans

from .helpers import convert_to_bgr, convert_to_lab, get_blurred_gray
from .segment_params import SegmentParams


def segment_beans(image: np.ndarray, params: SegmentParams, debug: bool = True) -> list[np.ndarray]:
    """Return list of contours for each bean using K-means clustering"""
    os.makedirs("steps", exist_ok=True)
    
    # Normalize contrast first
    normalized = _normalize_contrast(image)
    
    # Convert to LAB color space for better color clustering
    lab = convert_to_lab(normalized)
    
    # Apply clustering to segment beans from background
    clustered_mask = _cluster_segment(lab, params)
    
    # Apply morphological operations to clean up the mask
    opened = _open_foreground(clustered_mask, params.open_ksize)
    
    # Separate individual beans using distance transform and watershed
    separated_mask = _separate_beans(opened, params)
    
    if debug:
        result_vis = cv2.cvtColor(separated_mask, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(separated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_vis, contours, -1, (0, 255, 0), 2)
        
        # Create visualization of clustering result
        clustered_vis = _visualize_clusters(lab, params)
        
        for filename, img in {
            "1_normalized": normalized,
            "2_clustered": clustered_vis,
            "3_clustered_mask": cv2.cvtColor(clustered_mask, cv2.COLOR_GRAY2BGR),
            "4_opened": cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR),
            "5_separated": cv2.cvtColor(separated_mask, cv2.COLOR_GRAY2BGR),
            "6_result": result_vis,
        }.items():
            cv2.imwrite(f"steps/{filename}.png", img)

    contours, _ = cv2.findContours(
        separated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = [
        c for c in contours 
        if params.min_area <= cv2.contourArea(c) <= params.max_area
    ]

    return valid_contours


def segment_single_bean(image: np.ndarray, params: SegmentParams) -> list[np.ndarray]:
    """Segment single bean from training image using K-means clustering."""
    # Normalize contrast first
    normalized = _normalize_contrast(image)
    
    # Convert to LAB color space
    lab = convert_to_lab(normalized)
    
    # Apply clustering to segment bean
    clustered_mask = _cluster_segment(lab, params)
    
    # Apply morphological operations
    opened = _open_foreground(clustered_mask, params.open_ksize)

    # Find contours
    contours, _ = cv2.findContours(
        opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter by area and return the largest valid contour
    valid_contours = [
        c for c in contours if params.min_area <= cv2.contourArea(c) <= params.max_area
    ]

    return valid_contours


def _normalize_contrast(img_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE to L channel in Lab for gentle contrast normalization."""
    lab = convert_to_lab(img_bgr)
    brightness, green_red, blue_yellow = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_brightness = clahe.apply(brightness)
    lab_enhanced = cv2.merge([enhanced_brightness, green_red, blue_yellow])
    return convert_to_bgr(lab_enhanced)


def _preprocess_to_gray(image: np.ndarray) -> np.ndarray:
    image = _normalize_contrast(image)
    return get_blurred_gray(image)


def _binarize_to_foreground(image: np.ndarray) -> np.ndarray:
    _, threshold = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return threshold if threshold.mean() < 127 else cv2.bitwise_not(threshold)


def _cluster_segment(lab_image: np.ndarray, params: SegmentParams) -> np.ndarray:
    """
    Segment beans using K-means clustering on LAB color space.
    Returns a binary mask where white pixels represent beans.
    """
    h, w = lab_image.shape[:2]
    
    # Reshape image to a list of pixels (each pixel is a 3D LAB vector)
    pixels = lab_image.reshape(-1, 3).astype(np.float32)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=params.n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(pixels)
    
    # Reshape labels back to image shape
    label_image = labels.reshape(h, w)
    
    # Select which clusters represent beans
    bean_clusters = _select_bean_clusters(
        lab_image, label_image, kmeans.cluster_centers_, params
    )
    
    # Create binary mask: 255 for bean pixels, 0 for background
    mask = np.zeros((h, w), dtype=np.uint8)
    for cluster_id in bean_clusters:
        mask[label_image == cluster_id] = 255
    
    return mask


def _select_bean_clusters(
    lab_image: np.ndarray,
    label_image: np.ndarray,
    cluster_centers: np.ndarray,
    params: SegmentParams,
) -> list[int]:
    """
    Select which clusters represent beans (not background).
    Uses cluster size and brightness to identify bean clusters.
    """
    h, w = lab_image.shape[:2]
    total_pixels = h * w
    
    # Calculate cluster statistics
    cluster_stats = []
    for cluster_id in range(len(cluster_centers)):
        cluster_mask = (label_image == cluster_id)
        cluster_size = np.sum(cluster_mask)
        cluster_ratio = cluster_size / total_pixels
        
        # Get average L (brightness) value for this cluster
        cluster_pixels = lab_image[cluster_mask]
        avg_brightness = np.mean(cluster_pixels[:, 0]) if len(cluster_pixels) > 0 else 0
        
        cluster_stats.append({
            'id': cluster_id,
            'size': cluster_size,
            'ratio': cluster_ratio,
            'brightness': avg_brightness,
        })
    
    # Select bean clusters based on method
    if params.cluster_selection_method == "brightness":
        # Select clusters with medium brightness (beans are usually darker than white background,
        # but not as dark as shadows)
        sorted_by_brightness = sorted(cluster_stats, key=lambda x: x['brightness'])
        # Select clusters in the middle brightness range (beans are typically darker than white bg)
        bean_clusters = [
            stat['id'] for stat in sorted_by_brightness
            if 20 <= stat['brightness'] <= 85  # Reasonable brightness range for beans
        ]
        # If we filtered out too many, select all except the brightest (likely white background)
        if not bean_clusters or len(bean_clusters) == len(cluster_stats):
            brightest = max(cluster_stats, key=lambda x: x['brightness'])
            bean_clusters = [stat['id'] for stat in cluster_stats if stat['id'] != brightest['id']]
    else:  # size method (default)
        # Select clusters that are NOT the largest (likely background)
        # and not too small (likely noise)
        sorted_by_size = sorted(cluster_stats, key=lambda x: x['size'], reverse=True)
        largest_cluster = sorted_by_size[0]
        
        # Select all clusters except the largest one, but filter out very small ones
        bean_clusters = [
            stat['id'] for stat in sorted_by_size[1:]  # Skip largest (background)
            if stat['ratio'] > 0.01  # At least 1% of image (less restrictive)
        ]
        
        # If we filtered out everything, include all except the largest
        if not bean_clusters:
            bean_clusters = [stat['id'] for stat in sorted_by_size[1:]]
    
    # Ensure we have at least one cluster selected
    if not bean_clusters:
        # Fallback: select all clusters except the largest one
        largest_cluster = max(cluster_stats, key=lambda x: x['size'])
        bean_clusters = [stat['id'] for stat in cluster_stats if stat['id'] != largest_cluster['id']]
    
    return bean_clusters


def _visualize_clusters(lab_image: np.ndarray, params: SegmentParams) -> np.ndarray:
    """Create a visualization of the clustering result for debugging."""
    h, w = lab_image.shape[:2]
    pixels = lab_image.reshape(-1, 3).astype(np.float32)
    
    kmeans = KMeans(n_clusters=params.n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(pixels)
    label_image = labels.reshape(h, w)
    
    # Create colored visualization: each cluster gets a different color
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    
    for cluster_id in range(params.n_clusters):
        color = colors[cluster_id % len(colors)]
        vis[label_image == cluster_id] = color
    
    return vis


def _separate_beans(mask: np.ndarray, params: SegmentParams) -> np.ndarray:
    """
    Separate individual beans from connected regions using distance transform
    and watershed algorithm, with fallback to erosion-based separation.
    """
    # First, try to use connected components - if beans are already separated, this works
    num_labels, labels = cv2.connectedComponents(mask)
    
    # If we already have many separate components, use them directly
    if num_labels > 3:  # More than just background + a couple beans
        return mask
    
    # Otherwise, beans are touching - use watershed to separate them
    # Compute distance transform to find bean centers
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = np.max(dist_transform)
    
    if max_dist == 0:
        return mask
    
    # Find local maxima (bean centers) - use a lower threshold to get more markers
    local_maxima_thresh = float(max_dist * 0.25)  # Lower threshold = more markers
    
    # Find peaks
    _, sure_fg = cv2.threshold(dist_transform.astype(np.float32), local_maxima_thresh, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    
    # Create sure background by dilating
    kernel_size = max(3, params.open_ksize)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    sure_bg = cv2.dilate(mask, kernel, iterations=3)
    
    # Unknown region
    unknown = cv2.subtract(sure_bg.astype(np.int32), sure_fg.astype(np.int32))
    unknown = np.clip(unknown, 0, 255).astype(np.uint8)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg.astype(np.uint8))
    markers = markers + 1  # Background becomes 1
    markers[unknown == 255] = 0  # Unknown becomes 0
    
    # Apply watershed
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(mask_3ch, markers)
    
    # Create output mask
    output_mask = np.zeros_like(mask)
    output_mask[markers > 1] = 255  # All markers > 1 are beans
    
    # If watershed didn't create enough separations, try erosion-based approach
    num_separated = len(np.unique(markers[markers > 1]))
    if num_separated < 2:
        # Fallback: use erosion to separate touching beans
        output_mask = np.zeros_like(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Erode to separate beans
        eroded = cv2.erode(mask, kernel, iterations=2)
        num_labels, labels = cv2.connectedComponents(eroded)
        
        # For each separated component, dilate back but limit growth
        for label_id in range(1, num_labels):
            component = (labels == label_id).astype(np.uint8) * 255
            # Dilate but intersect with original mask to prevent overlap
            dilated = cv2.dilate(component, kernel, iterations=3)
            final_component = cv2.bitwise_and(dilated, mask)
            output_mask = cv2.bitwise_or(output_mask, final_component)
    
    return output_mask


def _open_foreground(
    image: np.ndarray,
    ksize: int,
) -> np.ndarray:
    k_shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, k_shape, iterations=1)


def get_contours(image: MatLike, single_bean: bool) -> list[np.ndarray]:
    if single_bean:
        return segment_single_bean(
            image,
            SegmentParams(
                min_area=1000,
                max_area=100000,
                open_ksize=5,
            ),
        )
    return segment_beans(
        image,
        SegmentParams(
            min_area=3000,
            max_area=30_000,
            open_ksize=7,
            n_clusters=3,  # Fewer clusters - just separate foreground from background
            cluster_selection_method="brightness",  # Better for beans vs white background
        ),
    )
