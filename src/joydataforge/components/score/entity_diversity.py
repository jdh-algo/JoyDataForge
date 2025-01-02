"""
This module defines the "EntityDiversity" class.

The file includes the following import statements:
import json
import numpy as np
from skbio.diversity import alpha_diversity
from sklearn.preprocessing import LabelEncoder
from src.joydataforge.utils.file import line_generator

The file also includes the following classes and functions:
- EntityDiversity class with methods: __init__, fetch_all_entities, shannon_entropy_and_simpson_index, get_diversity_metrics

To use this module, you can instantiate the EntityDiversity class with the appropriate parameters, call its 'get_diversity_metrics' method to perform statistical analysis on the extracted entities, and retrieve various diversity metrics such as Shannon entropy and Simpson index for both entities and labels.

"""
import json
import numpy as np
from skbio.diversity import alpha_diversity
from sklearn.preprocessing import LabelEncoder
from src.joydataforge.utils.file import line_generator


class EntityDiversity(object):
    def __init__(self, path: str = "", data: list = []) -> None:
        self.path = path
        self.data = data
        self.metric_shanno_entropy_entity = None
        self.metric_simpson_index_entity = None
        self.metric_shanno_entropy_label = None
        self.metric_simpson_index_label = None
        self.metric_entities_labels_nums = 0
        self.metric_unique_entities_nums = 0
        self.entities = None
        self.unique_entities = None
        self.unique_entities_labels = None
        self.entities_encoder = None
        self.label_encoder = None

        self.encoder = LabelEncoder()

    # Statistical analysis of the extracted entities
    async def fetch_all_entities(self, data_path: str = "", data: list = [], except_labels: list = [], exclude_labels: list = [],
                                 th: int = 1000):
        """
        Perform statistics on a given data file or process a given data list.
        """
        self.entities = []
        self.unique_entities = []
        self.entities_labels = []
        self.unique_entities_labels = []
        self.nums = 0
        if data_path:
            self.path = data_path
        if data:
            self.data = data
        if self.path:
            for line in line_generator(self.path):
                js = json.loads(line)
                self.nums += 1
                for key, val in js.items():
                    if except_labels:
                        if key in except_labels:
                            if exclude_labels:
                                if key in exclude_labels:
                                    continue
                            self.entities_labels.append(key)
                            self.entities.extend(val)
                    else:
                        if key not in exclude_labels:
                            self.entities_labels.append(key)
                            self.entities.extend(val)
        elif self.data:
            for line in self.data:
                if isinstance(line, str):
                    js = json.loads(line)
                else:
                    js = line
                self.nums += 1
                for key, val in js.items():
                    if except_labels:
                        if key in except_labels:
                            if exclude_labels:
                                if key in exclude_labels:
                                    continue
                            self.entities_labels.append(key)
                            self.entities.extend(val)
                    else:
                        if key not in exclude_labels:
                            self.entities_labels.append(key)
                            self.entities.extend(val)
        self.entities = [elem for elem in self.entities if isinstance(elem, str)]
        self.unique_entities = list(set(self.entities))
        self.unique_entities_labels = list(set(self.entities_labels))
        self.metric_unique_entities_nums = len(self.unique_entities)
        self.metric_entities_labels_nums = len(self.unique_entities_labels)

    # Shannon entropy and Simpson index
    async def shannon_entropy_and_simpson_index(self):
        self.entities_encoder = self.encoder.fit_transform(self.entities)
        counts_entities = np.bincount(self.entities_encoder)
        self.metric_shanno_entropy_entity = alpha_diversity('shannon', [counts_entities]).tolist()[0]
        self.metric_simpson_index_entity = alpha_diversity('simpson', [counts_entities]).tolist()[0]

        self.label_encoder = self.encoder.fit_transform(self.entities_labels)
        counts_labels = np.bincount(self.label_encoder)
        self.metric_shanno_entropy_label = alpha_diversity('shannon', [counts_labels]).tolist()[0]
        self.metric_simpson_index_label = alpha_diversity('simpson', [counts_labels]).tolist()[0]

    # Get all the set evaluation metrics for entity diversity
    async def get_diversity_metrics(self, data: list = [], path: str = "", except_labels: list = [], exclude_labels: list = []):
        await self.fetch_all_entities(data=data, data_path=path, except_labels=except_labels, exclude_labels=exclude_labels, th=100)
        await self.shannon_entropy_and_simpson_index()
        res = {}
        res["entity_shannon_entropy"] = self.metric_shanno_entropy_entity
        res["label_shannon_entropy"] = self.metric_shanno_entropy_label
        res["entity_simpson_index"] = self.metric_simpson_index_entity
        res["label_simpson_index"] = self.metric_simpson_index_label
        res["unique_entity_count"] = self.metric_unique_entities_nums
        res["unique_label_count"] = self.metric_entities_labels_nums
        res["total_data_count"] = self.nums

        return res
