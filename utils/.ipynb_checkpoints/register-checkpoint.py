"""Class registration utilities"""
from typing import Dict, Type, Any

class Registry:
    """Registry for dynamically loading classes"""
    
    def __init__(self):
        self._registry: Dict[str, Type] = {}
    
    def register(self, name: str, cls: Type):
        """Register a class"""
        self._registry[name] = cls
    
    def get_class(self, name: str) -> Type:
        """Get a registered class"""
        if name not in self._registry:
            raise ValueError(f"Class {name} not registered")
        return self._registry[name]

# Global registry instance
registry = Registry()

def register_class(alias: str):
    """Decorator to register a class"""
    def decorator(cls):
        registry.register(alias, cls)
        return cls
    return decorator