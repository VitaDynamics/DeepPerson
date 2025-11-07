"""DeepPerson: Person re-identification component for Vbot framework.

This package provides a simple API for person detection, embedding generation,
identity verification, and gallery-based person search.

Quick Start:
    >>> from deep_person import DeepPerson
    >>> dp = DeepPerson()
    >>> result = dp.represent("person.jpg")
    >>> is_same = dp.verify("person1.jpg", "person2.jpg")
"""

__version__ = "0.1.0"

from src.api import DeepPerson

__all__ = ["DeepPerson", "__version__"]
