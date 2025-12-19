import numpy as np
from collections import Counter
from typing import Optional, Any


class KNNClassifier:
    """
    K-Nearest Neighbors classifier (from scratch).
    
    Uses Euclidean distance and majority voting.
    """
    
    def __init__(self, cfg: Optional[Any] = None, k: int = 5):
        """
        Initialize KNN classifier.
        
        Args:
            cfg: Optional config object (e.g., cfg.model.knn)
            k: Number of neighbors (default: 5)
        """
        if cfg is not None:
            self.k = getattr(cfg, 'k', k)
        else:
            self.k = k
        
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """
        Lazy learning: just store the training data.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        print(f"   [KNN] Stored {len(X)} training samples.")
        
    def predict(self, X_test):
        """
        Predict labels for test data.
        """
        predictions = []
        
        # Loop through every test sample
        for i, x in enumerate(X_test):
            # 1. Calculate Euclidean Distance to ALL training points
            # (x - X_train) subtracts x from every row in X_train (broadcasting)
            # Square, Sum, Sqrt
            distances = np.sqrt(np.sum((x - self.X_train)**2, axis=1))
            
            # 2. Sort and get indices of the 'k' smallest distances
            k_indices = np.argsort(distances)[:self.k]
            
            # 3. Retrieve the labels for those 'k' neighbors
            k_nearest_labels = [self.y_train[idx] for idx in k_indices]
            
            # 4. Vote: Find the most common label
            # Counter.most_common(1) returns [(label, count)]
            most_common = Counter(k_nearest_labels).most_common(1)
            predicted_label = most_common[0][0]
            
            predictions.append(predicted_label)
            
        return np.array(predictions)