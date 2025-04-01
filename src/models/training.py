# src/models/training.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from typing import Dict
import numpy as np
import pandas as pd
from src.config import RANDOM_STATE

class ModelTrainer:
    """Handles model training and management"""
    
    @staticmethod
    def initialize_models(X_train: np.ndarray, y_train: pd.Series) -> Dict:
        """Initialize and configure machine learning models"""
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=150, 
                                                  max_depth=5, 
                                                  class_weight='balanced',
                                                  random_state=RANDOM_STATE),
            "Logistic Regression": LogisticRegression(max_iter=1000, 
                                                    class_weight='balanced', 
                                                    random_state=RANDOM_STATE),
            "SVM": SVC(probability=True, 
                      kernel='rbf', 
                      class_weight='balanced',
                      random_state=RANDOM_STATE),
            "KNN": KNeighborsClassifier(n_neighbors=7),
            "XGBoost": XGBClassifier(n_estimators=100, 
                                    max_depth=3, 
                                    scale_pos_weight=sum(y_train==0)/sum(y_train==1),
                                    random_state=RANDOM_STATE,
                                    eval_metric='logloss'),
            "Naive Bayes": GaussianNB()
        }
        return models

    @staticmethod
    def train_models(models: Dict, X_train: np.ndarray, 
                    y_train: pd.Series) -> Dict:
        """Train all models"""
        trained_models = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
        return trained_models