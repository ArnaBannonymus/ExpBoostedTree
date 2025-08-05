from abc import ABC, abstractmethod
import numpy as np

class BaseExplainer(ABC):
    @abstractmethod
    def explain_global(self, **kwargs) -> np.ndarray:
        ...
    @abstractmethod
    def explain_local(self, X: np.ndarray, **kwargs):
        ...
