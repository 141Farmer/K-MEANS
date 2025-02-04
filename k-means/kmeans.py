import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def initialize_centroids(X: np.ndarray, k: int) -> np.ndarray:
    """Randomly initializes k centroids from the dataset."""
    idx = np.random.choice(len(X), k, replace=False)
    return X[idx]

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assigns each point to the nearest centroid."""
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Recomputes centroids as the mean of assigned points."""
    return np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else np.random.rand(3) * 255 for i in range(k)])

def kmeans_clustering(X: np.ndarray, k: int, max_iters: int = 100) -> np.ndarray:
    """Performs K-Means clustering on pixel data."""
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels.reshape(X.shape[0]), centroids

def visualize_3d_clustering(img_path: str, sample_size: int = 1000, k: int = 2):
    """Visualizes K-Means clustering in 3D RGB space."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img.reshape(-1, 3)
    
    sampled_pixels = pixels[np.random.choice(len(pixels), sample_size, replace=False)]
    
    labels, centroids = kmeans_clustering(sampled_pixels, k)
    
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(sampled_pixels[:, 0], sampled_pixels[:, 1], sampled_pixels[:, 2], 
                c=sampled_pixels/255, marker='o', s=20)
    ax1.set_title('Original Points')
    ax1.set_xlabel('Red')
    ax1.set_ylabel('Green')
    ax1.set_zlabel('Blue')
    
    ax2 = fig.add_subplot(132, projection='3d')
    scatter = ax2.scatter(sampled_pixels[:, 0], sampled_pixels[:, 1], sampled_pixels[:, 2], 
                          c=labels, cmap='viridis', marker='o', s=20, zorder=1)
    ax2.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
                c='red', marker='x', s=400, linewidths=5, zorder=5)
    ax2.set_title('Clustered Points')
    ax2.set_xlabel('Red')
    ax2.set_ylabel('Green')
    ax2.set_zlabel('Blue')
    
    plt.tight_layout()
    plt.show()

def detect_skin(image_path: str, k: int = 2) -> np.ndarray:
    """Detects skin by clustering and selecting the brightest cluster."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img.reshape(-1, 3)
    
    labels, _ = kmeans_clustering(pixels, k)
    
    segmented = labels.reshape(img.shape[:2])
    
    # Identify the skin cluster using brightness
    mean_brightness = [np.mean(pixels[labels == i]) for i in range(k)]
    skin_cluster_idx = np.argmax(mean_brightness)  # Brightest cluster
    
    return (segmented == skin_cluster_idx).astype(np.uint8)

def visualize_results(img_path: str):
    """Displays original image, skin mask, and detected skin."""
    skin_mask = detect_skin(img_path)
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    skin_detected = img.copy()
    skin_detected[skin_mask == 0] = 0
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(skin_mask, cmap='gray')
    plt.title('Skin Mask')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(skin_detected)
    plt.title('Detected Skin')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# Usage
visualize_3d_clustering("hand.jpg")
visualize_results("hand.jpg")
