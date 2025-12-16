import numpy as np
from collections import Counter
from src.models.classifier import BirdClassifier
import copy



class KNearestNeighbors(BirdClassifier):
    @classmethod
    def run_multiple_file_KNN(cls, features):
        import random
        # Prepare feature matrix and labels
        X = np.array([[f['centroid'], f['bandwidth']] for f in features])
        y = np.array([f['class'] for f in features])

        # Shuffle and split into train/test (80% train, 20% test per class, ignore extras)
        class_to_indices = {}
        for idx, label in enumerate(y):
            class_to_indices.setdefault(label, []).append(idx)
        train_idx, test_idx = [], []
        for indices in class_to_indices.values():
            random.shuffle(indices)
            n = len(indices)
            n_train = int(n * 0.8)
            n_test = n - n_train
            # Only use even splits, ignore extras
            n_per_class = n_train + n_test
            indices = indices[:n_per_class]
            train_idx.extend(indices[:n_train])
            test_idx.extend(indices[n_train:n_train+n_test])
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        knn = cls(k=3)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        accuracy = np.mean(y_pred == y_test)
        print(f"KNN test accuracy (80/20 split, even per class): {accuracy:.2f}")
        return accuracy

    def __init__(self, k=3,num_classes = 6):  # odd n is better for binary comparison
        # Inherit from base model
        super().__init__(num_classes if num_classes is not None else 0)
        self.k = k
        # Initialize parameters
        self._observations = None  
        self._ground_truth = None  
        self._parameters = {
            "observations": None,
            "ground_truth": None
        }
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        # Infer num_classes from ground_truth if not set
        if self.num_classes == 0:
            self.num_classes = len(set(ground_truth))
        self._observations = copy.deepcopy(observations)
        self._ground_truth = copy.deepcopy(ground_truth)
        self._parameters = {
            "observations": copy.deepcopy(self._observations),
            "ground_truth": copy.deepcopy(self._ground_truth)
        }

    def fit(self, observations: np.ndarray,
            ground_truth: np.ndarray) -> None:
        # Copy the private parameters before modifying
        self._observations = copy.deepcopy(observations)
        self._ground_truth = copy.deepcopy(ground_truth)

        # Store parameters in a dictionary
        self._parameters = {
            "observations": copy.deepcopy(self._observations),
            "ground_truth": copy.deepcopy(self._ground_truth)
        }

    def predict(self, observations: np.ndarray):
        predictions = [self._predict_single(x) for x in observations]
        return np.array(predictions)

    def _predict_single(self, observation):
        # 1. Calculate Euclidean distance between observation and every other point
        observation = np.array(observation)
        distances = np.linalg.norm(
            self._parameters["observations"]
            - observation, axis=1
            )
        # 2. Sort the array of the distances and take the first k
        k_indices = np.argsort(distances)[:self.k]
        # 3. Check label of the first k points
        k_nearest_labels = [
            self._parameters["ground_truth"][i] for i in k_indices
                            ]
        # 4. Count and return the most common point
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
       