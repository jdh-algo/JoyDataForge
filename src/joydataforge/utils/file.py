"""
This module defines file and directory utilities for reading, writing, and processing files.

The file includes the following import statements:
- json
- random
- os
- glob
- logger from loguru
- Counter from collections

The file also includes the following classes and functions:
- read_file(file_path)
- ensure_dir(directory)
- get_random_file(dir_path)
- get_child_files(dir_path)
- file_exists_in_folder(filename, folder)
- get_random_line(file_path)
- read_file_line_by_line(filename)
- count_lines(filename)
- fuzzy_match_files(directory, pattern="*")
- line_generator(file_path)
- query_formated(query, reg=reg)
- find_most_frequent(lst)
- query_fetch_from_all_agent_training_dataset(path)
- convert_agent_training_dataset_2_standard_dataformat(path)

To use this module, you can import it into your project and call the functions to perform file and directory operations, such as reading files, ensuring directories exist, and processing text data.
"""

import glob
import json
import os
import random
from collections import Counter

from loguru import logger


def read_file(file_path):
    with open(file_path) as f:
        content = f.read()
    return content


def ensure_dir(directory):
    # Ensure the directory exists or create it
    os.makedirs(directory, exist_ok=True)


def get_random_file(dir_path):
    # Get all files in the directory and its subdirectories
    files = [os.path.join(root, name)
             for root, dirs, files in os.walk(dir_path)
             for name in files]
    # Randomly select a file from the list
    return random.choice(files)


def get_child_files(dir_path):
    files = [os.path.join(root, name)
             for root, dirs, files in os.walk(dir_path)
             for name in files]
    return files


def file_exists_in_folder(filename, folder):
    # Traverse all files in the folder
    for root, dirs, files in os.walk(folder):
        if filename in files:
            return True
    return False


def get_random_line(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return random.choice(lines)


def read_file_line_by_line(filename):
    with open(filename, 'r') as file:
        for line in file:
            yield line


def count_lines(filename):
    with open(filename, 'r') as f:
        for i, _ in enumerate(f, 1):
            pass
    return i


def fuzzy_match_files(directory, pattern="*"):
    """
    Fuzzy match all files in the directory.

    :param directory: The directory to search
    :param pattern: Fuzzy match pattern, default is '*' to match all files
    :return: List of matched file paths
    """
    # Build the search path
    search_pattern = os.path.join(directory, pattern)
    # Use glob.glob() for fuzzy matching
    matched_files = glob.glob(search_pattern)
    # Return the list of matched files
    return matched_files


def line_generator(file_path):
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            for line in file:
                yield line
    except GeneratorExit:
        try:
            with open(file_path, 'r', encoding="utf-8-sig") as file:
                for line in file:
                    yield line
        except GeneratorExit:
            try:
                with open(file_path, 'r', encoding="GB18030") as file:
                    for line in file:
                        yield line
            except GeneratorExit:
                try:
                    with open(file_path, 'r', encoding="latin-1") as file:
                        for line in file:
                            yield line
                except GeneratorExit:
                    try:
                        with open(file_path, 'r', encoding="latin-2") as file:
                            for line in file:
                                yield line
                    except GeneratorExit:
                        logger.error("file encoding error!")


reg = r'''[,.!?;:'"“”‘’\-—_(){}\[\]<>《》、，。！？；：‘’“”【】（）…\s]+'''


def query_formated(query, reg=reg):
    import re
    query = re.sub(reg, '', query)
    return query


def find_most_frequent(lst):
    # Use Counter to calculate the frequency of each element in the list
    count = Counter(lst)
    # most_common() method returns a list of the most common elements and their counts
    # We are only interested in the most common one, so we use index [0]
    most_common_element, frequency = count.most_common(1)[0]
    return most_common_element, frequency


@staticmethod
def query_fetch_from_all_agent_training_dataset(path):
    queries = []
    for line in line_generator(path):
        js = json.loads(line)
        if "current_conversation" in js:
            queries.append(js["current_conversation"]["user_query"])
        elif "user" in js:
            queries.append(js["user"])
    return queries


@staticmethod
def convert_agent_training_dataset_2_standard_dataformat(path):
    res = []
    i = 0
    for line in line_generator(path):
        temp = {}
        js = json.loads(line)
        if "current_conversation" in js:
            temp["query"] = js["current_conversation"]["user_query"]
            temp["history"] = js["history_conversations"] if "history_conversations" in js else []
            temp["target"] = js["target"]
            temp["input"] = js["input"]
            temp["id"] = js["id"]
        elif "user" in js:
            temp["query"] = js["user"]
            temp["history"] = []
            temp["target"] = js["target"]
            temp["input"] = js["input"]
            temp["id"] = js["id"]
        temp["id"] = i
        res.append(temp)
        i += 1
    with open(path + "_converted.json", "w", encoding="utf-8") as wf:
        wf.writelines("\n".join([json.dumps(one, ensure_ascii=False) for one in res]))
