"""Cross-camera association and global-track lifecycle."""

from .engine import AssociationEngine
from .models import AssociationConfig, Camera, ContractError, ExitEvent, Observation

__all__ = [
    "AssociationConfig",
    "AssociationEngine",
    "Camera",
    "ContractError",
    "ExitEvent",
    "Observation",
]
