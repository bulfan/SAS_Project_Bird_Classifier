import numpy as np
from sklearn.linear_model import LogisticRegression

class MultiLogRegression:
    def __init__(self, **kwargs):
        """
        Initialize the multinomial logistic regression classifier.
        kwargs are passed to sklearn's LogisticRegression.
        """
        self.model = LogisticRegression(solver='lbfgs', **kwargs)

    def fit(self, X, y):
        """
        Fit the model to the training data.
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict class labels for samples in X.
        X: Feature matrix (n_samples, n_features)
        Returns: Predicted class labels
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        """
        return self.model.score(X, y)

    def predict_proba(self, X):
        """
        Probability estimates for samples in X.
        """
        return self.model.predict_proba(X)
