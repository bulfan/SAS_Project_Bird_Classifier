import numpy as np
from collections import Counter
from models.classifier import BirdClassifier
import copy



class KNearestNeighbors(BirdClassifier):

    def __init__(self, k=3):  # odd n is better for binary comparison
        # Inherit from base model
        super().__init__()
        self.k = k
        # Initialize parameters
        self._observations = None  
        self._ground_truth = None  
        self._parameters = {
            "observations": None,
            "ground_truth": None
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
       