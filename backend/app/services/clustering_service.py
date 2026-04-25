import numpy as np
from sklearn.cluster import DBSCAN

def compute_clusters(points: list, eps_degrees: float = 0.005, min_samples: int = 3) -> list:
    """
    Groups heavily compacted marine debris inferences into spatial clusters for macro dashboard tracking.
    eps_degrees ~ 0.005 is roughly 500 meters at the equator.
    """
    if not points or len(points) < min_samples:
        return []
        
    coords = np.array([[p["lat"], p["lon"]] for p in points])
    
    clustering = DBSCAN(eps=eps_degrees, min_samples=min_samples).fit(coords)
    
    clusters = []
    unique_labels = set(clustering.labels_)
    
    for label in unique_labels:
        # -1 represents stray noisy macroplastics that didn't form a cluster
        if label == -1:
            continue 
        
        # Extract cluster center of mass
        cluster_mask = (clustering.labels_ == label)
        cluster_coords = coords[cluster_mask]
        
        center_lat = float(np.mean(cluster_coords[:, 0]))
        center_lon = float(np.mean(cluster_coords[:, 1]))
        density = int(np.sum(cluster_mask))
        
        clusters.append({
            "center": [center_lat, center_lon],
            "density": density
        })
        
    return clusters
