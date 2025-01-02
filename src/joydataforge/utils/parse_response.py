"""
This module defines the `ResponseParser` class for parsing JSON responses from GPT-based models.

The file includes the following import statements:
- json
- traceback
- ast
- logger from loguru

The file also includes the following classes and functions:
- class ResponseParser:
  - parse_response(self, text)
  - parse_gpt_data_generate_response(self, text)
  - parse_gpt_chatbot_data_res(self, text)

To use this module, you can create an instance of the `ResponseParser` class and call its methods to parse JSON responses from GPT models.
"""

import json
import traceback
import ast
from loguru import logger


class ResponseParser:

    def parse_response(self, text):
        try:
            text = json.loads(text)
            res = json.loads(text["choices"][0]["message"]["content"])
            return res
        except Exception as e:
            try:
                res = json.loads(text["choices"][0]["message"]["content"][8:-4])
                return res
            except Exception as e:
                try:
                    res = json.loads(text["choices"][0]["message"]["content"].split("\n")[1])
                    return res
                except Exception as e:
                    try:
                        if isinstance(text, dict):
                            res = text
                        else:
                            res = json.loads(text)
                        return res
                    except Exception as e:
                        traceback.print_exc()
                        logger.error("Parse error!")
                        logger.error("Text:", text)
                        logger.error("Type:", type(text))
                        return {"label": "", "reason": ""}

    def parse_gpt_data_generate_response(self, text):
        try:
            text = json.loads(text)
            res = text["choices"][0]["message"]["content"]
            return res
        except Exception as e:
            traceback.print_exc()
            logger.error("Parse error!")
            return ""

    def parse_gpt_chatbot_data_res(self, text):
        try:
            text = json.loads(text)
            res = text["choices"][0]["message"]["content"]

            # Attempt to parse the string as a Python object
            try:
                result = ast.literal_eval(res)
                if isinstance(result, list):
                    return result
            except:
                logger.error("Data is not a list!")

            return [x.strip() for x in res.split("\n") if len(x) > 20]

        except Exception as e:
            traceback.print_exc()
            logger.error("Parse error!")
            return []
