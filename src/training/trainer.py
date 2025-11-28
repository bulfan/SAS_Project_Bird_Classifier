"""
Training pipeline for bird sound classifier.
"""


class Trainer:
    """Trainer class for model training and evaluation."""

    def __init__(self, model, config: dict):
        """
        Initialize the trainer.

        Args:
            model: The classifier model to train.
            config: Training configuration dictionary.
        """
        self.model = model
        self.config = config

    def train(self, train_dataset, val_dataset=None):
        """
        Train the model.

        Args:
            train_dataset: Training dataset.
            val_dataset: Optional validation dataset.
        """
        raise NotImplementedError("Training not implemented")

    def evaluate(self, test_dataset):
        """
        Evaluate the model on test data.

        Args:
            test_dataset: Test dataset.

        Returns:
            Evaluation metrics.
        """
        raise NotImplementedError("Evaluation not implemented")
