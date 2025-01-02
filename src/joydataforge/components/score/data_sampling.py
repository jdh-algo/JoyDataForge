"""
This module defines the "DataSelect" and "DataSamplingAndEvaluation" classes.

The file includes the following import statements:
import os
import json
import traceback
from typing import List, Dict, Union, Optional, Any
import aiofiles
import numpy as np
from contextlib import asynccontextmanager
from src.joydataforge.components.filter.k_center_greedy import KCenterGreedy
from src.joydataforge.components.score.data_evaluation import DataEvaluation
from src.joydataforge.models.llm import EmbeddingModel
from loguru import logger

The file also includes the following classes and functions:
- DataSelectError exception class
- DataSelect class with methods: __init__, select_data, _random_select, _k_center_select
- DataSamplingAndEvaluation class with methods: __init__, initialize, close, sampling_session, sampling, _process_method, _save_results, get_selected_data, get_evaluation_results, merge_results

To use this module, you can instantiate the DataSamplingAndEvaluation class with the appropriate parameters, call its 'sampling' method to perform data sampling, and use the 'get_selected_data' and 'get_evaluation_results' methods to retrieve the sampled data and evaluation results.
"""

import os
import json
import traceback
from typing import List, Dict, Union, Optional, Any
import aiofiles
import numpy as np
from contextlib import asynccontextmanager

from src.joydataforge.components.filter.k_center_greedy import KCenterGreedy
from src.joydataforge.components.score.data_evaluation import DataEvaluation
from src.joydataforge.models.llm import EmbeddingModel
from loguru import logger


class DataSelectError(Exception):
    """Custom exception related to data selection."""
    pass


class DataSelect:
    """Data selection class, providing various data selection strategies."""

    SUPPORTED_METHODS = {"random", "k_center"}

    def __init__(self,
                 embeddings: np.ndarray,
                 method: str = "k_center",
                 already_selected: Optional[List[int]] = None) -> None:
        """
        Initialize the data selector.

        Args:
            embeddings: The embedding matrix of the data
            method: Selection method
            already_selected: List of indices of already selected samples

        Raises:
            ValueError: When the method is not in the list of supported methods
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported selection method: {method}. "
                             f"Supported methods are: {self.SUPPORTED_METHODS}")

        self.embeddings = embeddings
        self.method = method
        self.already_selected = already_selected or []
        self.k_center = KCenterGreedy(x=embeddings)

    async def select_data(self, num_samples: int, method: Optional[str] = None) -> np.ndarray:
        """Select data samples."""
        method = method or self.method
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported selection method: {method}")

        try:
            if method == "random":
                return await self._random_select(num_samples)
            return await self._k_center_select(num_samples)
        except Exception as e:
            raise DataSelectError(f"Error during data selection: {str(e)}")

    async def _random_select(self, num_samples: int) -> np.ndarray:
        """Randomly select samples."""
        return np.random.choice(
            self.embeddings.shape[0],
            size=num_samples,
            replace=False
        )

    async def _k_center_select(self, num_samples: int) -> np.ndarray:
        """Select samples using the k-center algorithm."""
        return await self.k_center.select_batch(
            model=None,
            already_selected=self.already_selected,
            n=num_samples
        )


class DataSamplingAndEvaluation():
    """Data sampling class, combining embeddings and evaluation metrics for data sampling."""

    def __init__(self,
                 data: List[Union[str, Dict[str, Any]]],
                 embedding: Optional[np.ndarray] = None,
                 select_nums: int = 1000,
                 select_methods: Optional[List[str]] = None,
                 eval_methods: Optional[List[str]] = None,
                 need_embedding: bool = True):
        """Initialize the data sampler."""
        self.data = data
        self.select_nums = select_nums
        self.select_methods = select_methods or ["k_center", "random"]
        self.eval_methods = eval_methods or ["vendi", "entity_diversity"]
        self.embedding_tool = None
        self.features = embedding
        self.need_embedding = need_embedding
        self._initialized = False
        # Used for entity_diversity evaluation, if the data has many fields, you need to specify the fields to use or the list of fields to skip
        self.exclude_labels = ["input", "history", "response", "id", "task", "from", "task_name", "label", "reason"]
        self.include_labels = []
        self.data_selector = None
        self.data_evaluation = None

    async def initialize(self) -> None:
        """Asynchronously initialize."""
        if self._initialized:
            return

        if self.features is None or self.need_embedding:
            self.embedding_tool = EmbeddingModel()
            self.features = await self.embedding_tool.get_all_embeddings(datas=self.data)

        self.data_selector = DataSelect(embeddings=self.features)
        self.data_evaluation = DataEvaluation(datas=self.data, embeddings=self.features)
        self._initialized = True

    async def close(self) -> None:
        """Clean up resources."""
        if self.embedding_tool:
            await self.embedding_tool.close()

    @asynccontextmanager
    async def sampling_session(self):
        """Sampling session context manager."""
        await self.initialize()
        try:
            yield self
        finally:
            await self.close()

    async def sampling(self,
                       output_dir: str = "",
                       num_samples: Optional[int] = None) -> Dict[str, Any]:
        """Perform data sampling."""
        if not self._initialized:
            await self.initialize()

        num_samples = num_samples or self.select_nums
        results = {}

        for method in self.select_methods:
            logger.info(f"Starting data selection using {method}, selecting {num_samples} samples")
            try:
                results[method] = await self._process_method(method, num_samples)
                if output_dir:
                    await self._save_results(output_dir, method, num_samples, results[method])

            except Exception as e:
                logger.info(f"Error processing method {method}: {str(e)}")
                traceback.print_exc()
                continue
        return results

    async def _process_method(self, method: str, num_samples: int) -> Dict[str, Any]:
        """Process a single selection method."""
        selected_indices = await self.data_selector.select_data(
            method=method,
            num_samples=num_samples
        )

        selected_data = [
            json.loads(item) if isinstance(item, str) else item
            for item in np.array(self.data)[selected_indices]
        ]
        selected_embeddings = self.features[selected_indices]

        eval_results = await self.data_evaluation.evaluate(
            datas=selected_data,
            methods=self.eval_methods,
            embeddings=selected_embeddings,
            exclude_labels=self.exclude_labels,
            include_labels=self.include_labels
        )

        return {
            "selected_data": selected_data,
            "selected_embeddings": selected_embeddings.tolist(),
            "evaluation": eval_results
        }

    async def _save_results(self,
                            output_dir: str,
                            method: str,
                            num_samples: int,
                            results: Dict[str, Any]) -> None:
        """Save results to a file."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"{method}_selected_{num_samples}.json"
        )

        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(results, ensure_ascii=False, indent=2))

    async def get_selected_data(self, method: str, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get the selected data.

        Args:
            method: Selection method
            results: Sampling results dictionary

        Returns:
            List of selected data

        Raises:
            KeyError: When the specified method is not found in the results
        """
        if method not in results:
            raise KeyError(f"Method {method} not found in results")
        return results[method]["selected_data"]

    async def get_evaluation_results(self, method: str, results: Dict[str, Any]) -> Dict[str, float]:
        """Get evaluation results.

        Args:
            method: Selection method
            results: Sampling results dictionary

        Returns:
            Dictionary of evaluation results

        Raises:
            KeyError: When the specified method is not found in the results
        """
        if method not in results:
            raise KeyError(f"Method {method} not found in results")
        return results[method]["evaluation"]

    @staticmethod
    async def merge_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple sampling results.

        Args:
            results_list: List of sampling results

        Returns:
            Merged results dictionary
        """
        merged = {}
        for results in results_list:
            for method, result in results.items():
                if method not in merged:
                    merged[method] = {
                        "selected_data": [],
                        "selected_embeddings": [],
                        "evaluation": {}
                    }

                merged[method]["selected_data"].extend(result["selected_data"])
                merged[method]["selected_embeddings"].extend(result["selected_embeddings"])

                # Merge evaluation results, taking the average
                for eval_method, score in result["evaluation"].items():
                    if eval_method not in merged[method]["evaluation"]:
                        merged[method]["evaluation"][eval_method] = []
                    merged[method]["evaluation"][eval_method].append(score)

        # Calculate the average of the evaluation metrics
        for method in merged:
            for eval_method in merged[method]["evaluation"]:
                scores = merged[method]["evaluation"][eval_method]
                merged[method]["evaluation"][eval_method] = sum(scores) / len(scores)

        return merged
