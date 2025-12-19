from sklearn.ensemble import RandomForestClassifier
from typing import Optional, Any


class RFClassifier:
    """
    Random Forest classifier (sklearn wrapper).
    """
    
    def __init__(self, cfg: Optional[Any] = None, 
                 n_estimators: int = 100, 
                 max_depth: Optional[int] = None):
        """
        Initialize Random Forest classifier.
        
        Args:
            cfg: Optional config object (e.g., cfg.model.random_forest)
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
        """
        if cfg is not None:
            n_estimators = getattr(cfg, 'n_estimators', n_estimators)
            max_depth = getattr(cfg, 'max_depth', max_depth)
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
    def fit(self, X, y):
        print(f"   [RF] Training Random Forest on {len(X)} samples...")
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Returns the feature importances (scores) from the trained model."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return None