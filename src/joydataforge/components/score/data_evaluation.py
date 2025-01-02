"""
This module defines the "DataEvaluation" class for evaluating the quality of data.

The file includes the following import statements:
- numpy as np: for numerical computations
- typing: for type hints
- joydataforge.components.score.vendi_scores: for Vendi score calculation
- src.joydataforge.components.score.entity_diversity: for entity diversity calculation
- loguru: for logging purposes

The file also includes the following classes and functions:
- DataEvaluation: a class for evaluating the quality of data using various methods

To use this module, you can create an instance of the DataEvaluation class and call its evaluate method.
"""
import numpy as np
from typing import List
from src.joydataforge.components.score.vendi_scores import VendiScore
from src.joydataforge.components.score.entity_diversity import EntityDiversity
from loguru import logger


class DataEvaluation(object):
    def __init__(self, datas: List = [], embeddings: np.ndarray = None, methods: List = ["vendi", "entity_diversity"]) -> None:
        self.datas = datas
        self.embeddings = embeddings
        self.vendi_tool = VendiScore(embeddings=embeddings)
        self.entity_tool = EntityDiversity()
        self.methods = methods

    async def evaluate(self, datas: List = [], methods: List = [], embeddings: np.ndarray = None, include_labels: List = [],
                       exclude_labels: List = []):
        metrics = {}
        if methods:
            self.methods = methods
        if embeddings.any():
            self.embeddings = embeddings
        if datas:
            self.datas = datas
        for method in self.methods:
            logger.info(f"Starting data evaluation, selected method: {method}")
            if method == "vendi":
                metrics["vendi_score"] = float(await self.vendi_tool.score(self.embeddings))
            elif method == "entity_diversity":
                metrics.update(await self.entity_tool.get_diversity_metrics(data=self.datas, except_labels=include_labels,
                                                                            exclude_labels=exclude_labels))
            else:
                logger.info(f"Please implement the data diversity evaluation method! The given method name is: {method}")
        return metrics
