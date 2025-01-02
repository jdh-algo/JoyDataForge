"""
This module defines the utilities for dialog sampling and processing.

The file includes the following import statements:
- random

The file also includes the following classes and functions:
- dialog_sampling_by_round(dialogs, threshold, front_or_back="back")
- find_last_patient_sentence(sentences, index)
- dialog_sampling(dialogs)

To use this module, you can import it into your project and call the functions to sample dialogs based on different criteria.
"""

import random

random.seed(0)


def dialog_sampling_by_round(dialogs, threshold, front_or_back="back"):
    """
    Sample dialogs by round with options to select from the back, front, or randomly.

    Parameters:
    - dialogs: List or string of dialog lines.
    - threshold: The round threshold for sampling.
    - front_or_back: Sampling direction, options are 'back', 'front', or 'random'.

    Returns:
    - cur_query: The current query line.
    - A string of sampled dialog lines.
    """

    if isinstance(dialogs, str):
        dialogs = dialogs.split("\n")
    if not threshold:
        return dialogs
    res = []

    threshold = 2 * threshold
    round = 0
    if front_or_back == "back":
        dialogs = dialogs[::-1]
    elif front_or_back == "random":
        dialogs = dialog_sampling(dialogs)[::-1]
    current_role = ""
    old_role = ""
    for one in dialogs:
        one = one.strip()
        if not one:
            continue
        if one.startswith("患者") or one.startswith("用户"):
            current_role = "患者"
        else:
            current_role = "医生"
        if current_role != old_role:
            round += 1
            old_role = current_role
        if front_or_back in ["back", "random"]:
            res.insert(0, one)
            if round > threshold and current_role == "患者":
                break
        else:
            res.append(one)
            if round > threshold and current_role == "患者":
                break
    cur_query = ""
    for one in res.copy()[::-1]:
        res.remove(one)
        if one.startswith("患者") or one.startswith("用户"):
            cur_query = one
            break
    return cur_query, "\n".join(res)


def find_last_patient_sentence(sentences, index):
    """
    Find the last sentence spoken by the patient before and after a given index.

    Parameters:
    - sentences: List of sentences.
    - index: The index to start searching from.

    Returns:
    - The index of the last patient sentence found, or None if not found.
    """
    last_patient_before = None
    last_patient_after = None

    for i in range(index, -1, -1):
        if sentences[i].strip().startswith("用户") or sentences[i].strip().startswith("患者"):
            while i > 0 and sentences[i - 1].startswith("用户"):
                i -= 1
            last_patient_before = i
            break

    for i in range(index, len(sentences)):
        if sentences[i].startswith("用户"):
            while i < len(sentences) - 1 and (
                    sentences[i + 1].strip().startswith("用户") or sentences[i + 1].strip().startswith("患者")):
                i += 1
            last_patient_after = i
            break

    if last_patient_before is not None and last_patient_after is not None:
        return last_patient_before if (index - last_patient_before) <= (last_patient_after - index) else last_patient_after
    elif last_patient_before is not None:
        return last_patient_before
    elif last_patient_after is not None:
        return last_patient_after

    return None


def dialog_sampling(dialogs):
    """
    Randomly sample a dialog, ensuring it includes the last complete patient sentence.

    Parameters:
    - dialogs: List of dialog dictionaries or strings.

    Returns:
    - A list of dialog lines up to the last patient sentence found.
    """
    index = random.randint(0, len(dialogs) - 1)
    if isinstance(dialogs[0], str):
        index_format = find_last_patient_sentence(sentences=dialogs, index=index)
    else:
        sentences = []
        for dialog in dialogs:
            if dialog["role"] == "usr":
                sentences.append(f"用户：{dialog['content']}")
            else:
                sentences.append(f"医生：{dialog['content']}")
        index_format = find_last_patient_sentence(sentences=sentences, index=index)
    if index_format is not None:
        return dialogs[:index_format + 1]
    else:
        return []
