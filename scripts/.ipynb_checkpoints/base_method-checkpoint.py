"""Base method class for all methods"""
from abc import ABC, abstractmethod

class BaseMethod(ABC):
    """Abstract base class for all methods"""
    
    @abstractmethod
    def __init__(self, dataset, args):
        """Initialize the method"""
        pass
    
    @abstractmethod
    def run(self):
        """Run the method on the dataset"""
        pass