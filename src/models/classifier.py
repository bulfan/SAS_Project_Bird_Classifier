"""
Bird species classifier model.
"""


class BirdClassifier:
    """Base classifier for bird species identification."""

    def __init__(self, num_classes: int):
        """
        Initialize the classifier.

        Args:
            num_classes: Number of bird species to classify.
        """
        self.num_classes = num_classes
        self.model = None

    def build(self):
        """Build the classification model architecture."""
        raise NotImplementedError("Model building not implemented")

    def predict(self, features):
        """
        Predict bird species from audio features.

        Args:
            features: Preprocessed audio features.

        Returns:
            Predicted species class.
        """
        raise NotImplementedError("Prediction not implemented")

    def save(self, path: str):
        """Save the model to disk."""
        raise NotImplementedError("Model saving not implemented")

    def load(self, path: str):
        """Load the model from disk."""
        raise NotImplementedError("Model loading not implemented")
