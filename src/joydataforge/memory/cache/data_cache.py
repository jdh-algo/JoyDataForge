"""
This module defines the `DataCache` class, which is used to construct a cache of already generated data to prevent duplication, improve production efficiency, and save costs.

The file includes the following import statements:
- os
- json
- re
- traceback
- glob
- AnyStr, List, Optional, Any from typing
- defaultdict from collections
- logger from loguru
- line_generator from src.joydataforge.utils.file

The file also includes the following class and methods:
- DataCache
  - __init__
  - add_cache
  - cache_build
  - cache_load
  - save_cache

To use this module, you can instantiate the `DataCache` class with appropriate arguments and use its methods to build, add to, load, and save the cache.
"""

import os
import json
import re
import traceback
import glob
from typing import AnyStr, List, Optional, Any
from collections import defaultdict
from loguru import logger
from src.joydataforge.utils.file import line_generator


class DataCache(object):
    """Constructs a cache of already generated data to prevent duplication, improve production efficiency, and save costs.

    Args:
        object (_type_): _description_
    """

    def __init__(self, data_cache_read_path: Optional[AnyStr] = None, flag: Optional[AnyStr] = None, sub_path: Optional[AnyStr] = None,
                 recursive: Optional[bool] = True, key_list: Optional[List] = []):
        """Initializes the DataCache object.

        Args:
            data_cache_read_path (Optional[AnyStr], optional): Path to read data cache from. Defaults to None.
            flag (Optional[AnyStr], optional): File type, such as json, text, xlsx. Defaults to None.
            sub_path (Optional[AnyStr], optional): Subdirectory path. Defaults to None.
            recursive (Optional[bool], optional): Whether to recursively scan directories. Defaults to True.
            key_list (Optional[List], optional): List of keys used as fields for the cache's key. Defaults to [].
        """
        self.key_list = key_list
        self.path = data_cache_read_path
        self.flag = flag
        self.cache = defaultdict()
        if self.path:
            self.cache_build(sub_path=sub_path, recursive=recursive)

    def add_cache(self, query: AnyStr, target: Any):
        """Adds a query-target pair to the cache if it does not already exist.

        Args:
            query (AnyStr): The query string.
            target (Any): The target data associated with the query.

        Returns:
            bool: Returns False if the query was added, True if it already existed in the cache.
        """
        reg = r'''[,.!?;:'"“”‘’\-—_(){}\[\]<>《》、，。！？；：‘’“”【】（）…\s]+'''
        query = re.sub(reg, '', query)
        if query not in self.cache:
            self.cache[query] = target
            return False
        return True

    def cache_build(self, sub_path: Optional[str] = None, recursive: Optional[bool] = True, tag: Optional[AnyStr] = None):
        """Builds the cache by reading data from files.

        Args:
            sub_path (Optional[str], optional): Subdirectory path. Defaults to None.
            recursive (Optional[bool], optional): Whether to recursively scan directories. Defaults to True.
            tag (Optional[AnyStr], optional): Specific tag to extract from the data. Defaults to None.
        """
        logger.info(f"The root path of cached data is: {self.path}, sub-path is: {sub_path}")
        if isinstance(self.path, str):
            self.path = [self.path]
        for data_path in self.path:
            if sub_path:
                data_path = os.path.join(data_path, sub_path)
            files = glob.glob(data_path + "/*" + self.flag, recursive=recursive)
            for file in files:
                logger.info(file)
                for line in line_generator(file):
                    try:
                        js = json.loads(line)
                        if self.key_list:
                            input = "".join([js.get(k, "") for k in self.key_list])
                        else:
                            input = js["input"] if "input" in js else ""
                        if "chatbot" in file:
                            input = input + js["report"] if "report" in js else ""
                        if tag:
                            target = js[tag] if tag in js else ""
                        else:
                            target = js
                        if target:
                            self.add_cache(input, target)
                    except Exception as e:
                        traceback.print_exc()
                        logger.error("error, file:", file)
                        logger.error("error, line:", line)
                        continue
                logger.info(f"{file} processing completed!")

    def cache_load(self):
        """Loads the cache.

        Returns:
            dict: The cache dictionary.
        """
        return self.cache

    def save_cache(self, save_path: AnyStr):
        """Saves the cache to a specified file path.

        Args:
            save_path (AnyStr): The path where the cache should be saved.
        """
        with open(save_path, mode="w", encoding="utf-8") as wf:
            json.dump(obj=self.cache, fp=wf, ensure_ascii=False, indent=1)
