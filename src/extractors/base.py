"""Common interface for all relation extractors."""

from abc import ABC, abstractmethod

from src.schemas.prediction import Prediction
from src.schemas.text import CandidatePair


class BaseExtractor(ABC):
    """
    Abstract base extractor that all extractors (Rules, DistilBERT, LLM) must implement.
    All extractors operate on identical CandidatePair instances and return validated Prediction objects.
    """

    name: str
    version: str

    @abstractmethod
    def predict(self, pairs: list[CandidatePair]) -> list[Prediction]:
        """
        Extract relations from a list of candidate pairs.

        Args:
            pairs: List of CandidatePair instances.

        Returns:
            List of Prediction instances matching the candidate pairs.
        """
        pass
