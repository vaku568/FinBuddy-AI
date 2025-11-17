# base_model.py
import joblib
import os
from abc import ABC, abstractmethod

class BaseFinancialModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_trained = False
        self.feature_names = []

    @abstractmethod
    def train(self, X_train, y_train):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, X_test, y_test):
        raise NotImplementedError

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[SAVED] {path}")

    @staticmethod
    def load(path: str):
        return joblib.load(path)
