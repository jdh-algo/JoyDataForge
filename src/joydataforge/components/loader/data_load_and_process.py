"""
This module defines the "DataLoadAndProcess" class.

The file includes the following import statements:
import os
import json
import glob
import pandas as pd
from typing import Any, AnyStr, Optional, Tuple, AsyncGenerator
from src.joydataforge.utils.dialog_sample_by_round import dialog_sampling_by_round
from loguru import logger
from src.joydataforge.components.filter.mini_hash import MiniHashDedup

The file also includes the following classes and functions:
- DataLoadAndProcess class
- read_file_line_by_line async method
- read_data async method
- dialog_data_process async method
- check_encoding static method

To use this module, you can instantiate the DataLoadAndProcess class with the appropriate parameters and call its methods to read and process data files, including deduplication and dialog data processing.
"""

import os
import json
import glob
import pandas as pd

from typing import Any, AnyStr, Optional, Tuple, AsyncGenerator
from src.joydataforge.utils.dialog_sample_by_round import dialog_sampling_by_round
from loguru import logger
from src.joydataforge.components.filter.mini_hash import MiniHashDedup

CODER_LIST = ["utf-8", "utf-8-sig", "GB18030", "latin-1", "latin-2"]


class DataLoadAndProcess:
    def __init__(self, path: AnyStr, task: AnyStr, file_type: Optional[AnyStr] = "", data_type: Optional[AnyStr] = "str",
                 is_need_dedup: bool = True) -> None:
        self.task = task
        self.data_path = path
        self.file_type = file_type
        self.data_type = data_type
        self.mini_hash = None
        self.is_need_dedup = is_need_dedup
        if self.is_need_dedup:
            self.mini_hash = MiniHashDedup()

        logger.info(
            f"Task type: {self.task}, Data loading path: {self.data_path}, Specified file type: {self.file_type}, Data type: {self.data_type}")

    async def read_file_line_by_line(self, file_path: AnyStr) -> AsyncGenerator[Any, Any]:
        """Read file content and return it line by line."""
        if not await self.check_encoding(file_path):
            raise ValueError("Invalid file encoding")

        if self.file_type in ["txt", "json", "jsonl"]:
            if self.data_type == "str":
                with open(file_path, 'r', encoding=CODER_LIST[0]) as file:
                    for line in file:
                        yield line
            elif self.data_type == "list":
                with open(file_path, 'r', encoding=CODER_LIST[0]) as file:
                    for line in json.load(file):
                        yield json.dumps(line, ensure_ascii=False)
        elif self.file_type in ["xlsx", "csv"]:
            if self.file_type == "xlsx":
                lines = pd.read_excel(file_path, chunksize=1)
            else:
                lines = pd.read_csv(file_path, chunksize=1)
            for chunk in lines:
                yield chunk.to_json(orient="records", force_ascii=False)[1:-1]

    async def read_data(self) -> AsyncGenerator[Any, Any]:
        """Read data from the specified path."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path does not exist!\n{self.data_path}")

        if os.path.isfile(self.data_path):
            file_path = self.data_path
            if not self.file_type:
                _, ext = os.path.splitext(file_path)
                self.file_type = ext[1:]
            async for line in self.read_file_line_by_line(file_path):
                if self.is_need_dedup:
                    async for one in self.mini_hash.filter_processing(inputs=[json.loads(line)],
                                                                      need_return_dup_lines=not self.is_need_dedup, batch_size=1):
                        yield json.dumps(one, ensure_ascii=False)
        else:
            if not self.file_type:
                raise ValueError("File type must be specified when reading from a directory")

            file_list = glob.glob(os.path.join(self.data_path, f"**/*.{self.file_type}"), recursive=True)
            for file_path in file_list:
                async for line in self.read_file_line_by_line(file_path):
                    if self.is_need_dedup:
                        async for one in self.mini_hash.filter_processing(inputs=[json.loads(line)],
                                                                          need_return_dup_lines=not self.is_need_dedup, batch_size=1):
                            yield json.dumps(one, ensure_ascii=False)

    async def dialog_data_process(self, one_data: AnyStr, key: Optional[AnyStr] = "dialogs", type: Optional[AnyStr] = "random",
                                  round_th: Optional[int] = 2) -> Tuple[AnyStr, AnyStr]:
        """Process dialog data.

        Args:
            one_data (AnyStr): Input dialog data in JSON string format.
            key (Optional[AnyStr], optional): The key to retrieve dialog data, default is "dialogs".
            type (Optional[AnyStr], optional): Method for extracting dialog history: front, random, back.
            round_th (Optional[int], optional): Number of dialog rounds. Defaults to 2.

        Returns:
            Tuple[AnyStr, AnyStr]: Processed dialog data.
        """

        one_data_js = json.loads(one_data)
        dialogs = one_data_js.get(key, [])
        current_query, dialog_his = dialog_sampling_by_round(dialogs=dialogs, threshold=round_th, front_or_back=type)
        return current_query, dialog_his

    @staticmethod
    async def check_encoding(file_path: AnyStr) -> bool:
        """Check file encoding."""
        for coder in CODER_LIST:
            try:
                with open(file_path, 'r', encoding=coder) as file:
                    file.read()
                return True
            except (UnicodeDecodeError, FileNotFoundError):
                continue
        raise ValueError("Unable to read file with provided encodings")
