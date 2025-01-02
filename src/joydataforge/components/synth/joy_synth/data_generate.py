"""
This module is designed to facilitate the generation of data through various strategies using machine learning models. It defines the JoyDataGenerate class, which provides a framework for generating data asynchronously with methods that incorporate model voting, synthesis, case rewriting, and more. This class is especially useful in scenarios where you need to generate or augment datasets by leveraging machine learning models to simulate or predict data points.

The file includes the following import statements:
import datetime
import copy
import os
import json
import re
import time
import traceback
import pandas as pd
import random
import asyncio
from asyncio import Lock
import aiofiles
from tqdm import tqdm
from typing import Any, List, Dict, AnyStr, DefaultDict, Optional
from pydantic import BaseModel, Field
from loguru import logger
from src.joydataforge.utils.file import line_generator
from src.joydataforge.utils.parse_response import ResponseParser
from src.joydataforge.components.loader.data_load_and_process import DataLoadAndProcess
from src.joydataforge.memory.cache.data_cache import DataCache
from src.joydataforge.utils.file import find_most_frequent
from src.joydataforge.components.synth.joy_synth.wizard import create_wizard_prompt, convert_query_format

The file also includes the following classes and functions:
- JoyDataGenerate class with methods: __init__, process_model, generate, generate_one, generate_one_by_model_vote, generate_one_by_vote_and_pipeline, generate_one_by_synthsis, generate_one_by_case_rewrite, generate_one_by_all_intention, is_in_cache, sampling_from_data_with_all_intentions, find_his, generate_one_by_wizard_evolution, generate_one_by_dpo_scoring

To use this module, you can instantiate the JoyDataGenerate class with the appropriate configuration, and call the 'generate' method to asynchronously generate data using various strategies such as model voting, synthesis, case rewriting, and more.

"""

import datetime
import copy
import os
import json
import re
import time
import traceback
import pandas as pd
import random
import asyncio
from asyncio import Lock
import aiofiles
from tqdm import tqdm
from typing import Any, List, Dict, AnyStr, DefaultDict, Optional
from pydantic import BaseModel, Field
from loguru import logger
from src.joydataforge.utils.file import line_generator
from src.joydataforge.utils.parse_response import ResponseParser
from src.joydataforge.components.loader.data_load_and_process import DataLoadAndProcess
from src.joydataforge.memory.cache.data_cache import DataCache
from src.joydataforge.utils.file import find_most_frequent

from loguru import logger
from src.joydataforge.components.synth.joy_synth.wizard import create_wizard_prompt, convert_query_format


class JoyDataGenerate(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }
    model: Any
    output_file: str
    prompt: Optional[str] = Field("")
    max_works: Optional[int] = Field(default=10)
    data_processor: Optional[DataLoadAndProcess] = None
    model_list: List = Field(default_factory=list)
    method: str = Field(default="default")
    have_datas: bool = Field(default=False)
    data_path: str = Field(default="")
    task_name: str = Field(default="")
    from_where: str = Field(default="")
    caches: Optional[DataCache] = None
    threshold: int = Field(default=100)
    threshold_low: int = Field(default=0)
    insert_key_list: List = Field(default_factory=list)
    batch_size: int = Field(default=2)
    prompt_post_pipline: str = Field(default="")
    need_post_pipeline_label_list: List = Field(default_factory=list)
    wizard_evolution_round:int=Field(default=2)
    use_history:bool=Field(default=False)
    

    def __init__(self, **data):
        super().__init__(**data)
        self.lock = Lock()  # Use an asynchronous lock
        self.res_templete = json.dumps({"label":"", "score":"", "reason":"","input":"","target":"", "from":"","task_name":""}, ensure_ascii=False)+"\n"
        self.response_parser = ResponseParser()
        logger.info(f"Basic configuration information for data generation: {self.model_dump}")

    # Create tasks for all models
    async def process_model(self, model:Any, prompt:AnyStr) -> Dict[str, Any]:
        res = {}
        for i in range(10, 0, -1):
            try:
                # The model supports asynchronous calls
                result = await model.chat(prompt)
                res_temp = self.response_parser.parse_response(result)
                label = ""
                reason = ""
                scores = ""
                if res_temp:
                    label = res_temp.pop("label") if "label" in res_temp else ""
                    reason = res_temp.pop("reason") if "reason" in res_temp else ""
                    scores = res_temp.pop("reward_model_scores") if "reward_model_scores" in res_temp else []
                res[model.model_name + "_label"] = label
                res[model.model_name + "_reason"] = reason
                res[model.model_name + "_scores"] = scores
                res[model.model_name + "_res"] = res_temp if res_temp else []
                return res
            except Exception as e:
                logger.error(f"Model {model.model_name} error: {e}")
                logger.error(traceback.print_exc())
                continue
        res[model.model_name + "_label"] = ""
        res[model.model_name + "_reason"] = ""
        res[model.model_name + "_scores"] = []
        return res

    async def generate(self):
        """Generate data asynchronously"""
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
        # Batch processing function
        async def process_batch(batch):
            tasks = []
            for line in batch:
                if self.method == "default":
                    tasks.append(self.generate_one(line))
                elif self.method == "vote":
                    tasks.append(self.generate_one_by_vote(line))
                elif self.method == "vote_and_pipeline":
                    tasks.append(self.generate_one_by_vote_and_pipeline(line))
                elif self.method == "case_rewrite":
                    tasks.append(self.generate_one_by_case_rewrite(line))
                elif self.method == "synthesis_directly":
                    tasks.append(self.generate_one_by_synthesis(line)),
                elif self.method == "all_query_intentions":
                    tasks.append(self.generate_one_by_all_intention(line))
                elif self.method == "wizard_evolution":
                    tasks.append(self.generate_one_by_wizard_evolution(line))
                elif self.method == "dpo_reward_model_scoring":
                    tasks.append(self.generate_one_by_dpo_scoring(line))
                else:
                    logger.info("Please specify the given data synthesis method, or implement the data synthesis algorithm you need~")
                # ... You can add your own methods
            return await asyncio.gather(*tasks)

        # Write results
        async def write_results(results, file):
            async with aiofiles.open(file, 'a', encoding="utf-8") as f:
                for result in results:
                    await f.write(result)

        # Main processing logic
        current_batch = []
        index = 0
        
        async with aiofiles.open(self.output_file, 'w', encoding="utf-8") as w:
            if self.data_processor:
                pbar = tqdm(desc="Processing data", 
                            unit="lines",
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
                            )
                async for line in self.data_processor.read_data():
                    if self.threshold and index >= self.threshold:
                        break
                    if self.threshold_low and index <= self.threshold_low:
                        index += 1
                        continue
                        
                    current_batch.append(line)
                    if len(current_batch) >= self.batch_size:
                        results = await process_batch(current_batch)
                        await write_results(results, self.output_file)
                        
                        current_batch = []
                    index += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        'batch_size': len(current_batch),
                        'total_processed': index
                    })
                # Process remaining data
                if current_batch:
                    results = await process_batch(current_batch)
                    await write_results(results, self.output_file)

        # # Write statistical results
        # async with aiofiles.open(f"{self.output_file}_{self.model.model_name}_statistics.json", 'w', encoding="utf-8") as wf:
        #     await wf.write(json.dumps(self.statistics_res, ensure_ascii=False, indent=1))

    async def generate_one(self, line: str) -> str:
        """Asynchronous single-model generation"""
        try:
            js = json.loads(line)
            query = js.get('input', "")
            his = js.get('history', [])
            if isinstance(query, list):
                query = "\n".join(query) 
            # Check cache using asynchronous lock
            async with self.lock:
                flag, target = self.is_in_cache(query, self.caches.cache)
                if flag:
                    return json.dumps(target, ensure_ascii=False) + '\n'
            # Asynchronous model call
            if self.insert_key_list:
                input_temp = {k: js.get(k, "") for k in self.insert_key_list}
                prompt = self.prompt.format(**input_temp)
            else:
                prompt = self.prompt.format(input=query, history=his)
            res = await self.process_model(model=self.model, prompt=prompt)
            if res:
                js.update(res)
                js["from"] = self.from_where 
                js["task"] = self.task_name                
                async with self.lock:
                    self.caches.add_cache(query, js)                    
            return json.dumps(js, ensure_ascii=False) + '\n'
            
        except Exception as e:
            logger.error(traceback.print_exc())
            return self.res_templete

    async def generate_one_by_model_vote(self, line: str):
        """
        Asynchronous multi-model voting generation"
        """
        try:
            js = json.loads(line)
            query = js.get('input')
            his = js.get('history')
            prompt = self.prompt.format(input=query, history=his)
            
            if isinstance(query, list):
                query = "\n".join(query)

            # Check cache using asynchronous lock
            async with self.lock:
                flag, target = self.is_in_cache(query+his, data_cache=self.caches.cache)
                if flag:
                    return json.dumps(target, ensure_ascii=False) + '\n'

            begin = time.time()
            labels = []
            tasks = [self.process_model(model, prompt=prompt) for model in self.model_list]
            model_results = await asyncio.gather(*tasks)

            # Aggregate results
            for result in model_results:
                if result:
                    for k, v in result.items():
                        if "_label" in k:
                            labels.append(result["label"])
                            break
                    js.update(result)
     
            # Find the most voted label
            label, fre = find_most_frequent(labels)
            js["label"] = label
            js["from"] = self.from_where
            js["task"] = self.task_name

            # Cache the result using an asynchronous lock
            async with self.lock:
                self.caches.add_cache(query+his, js)

            output_line = json.dumps(js, ensure_ascii=False) + '\n'
            logger.info(f"single time  use ：{time.time()-begin}")
            return output_line
        except Exception as e:
            logger.error(traceback.print_exc())
            return self.res_templete

    async def generate_one_by_vote_and_pipeline(self, line: str, needed_label_list: List[str] = []):
        """ synth one data by multi-model vote and a super model for post processing
        """
        try:
            js = json.loads(line)
            query = js.get('input')
            his = js.get('history')
            prompt = self.prompt.format(input=query, history=his)
            prompt_post_processing = self.prompt_post_pipline.format(input=query, history=his)
            if isinstance(query, list):
                query = "\n".join(query)
            async with self.lock:
                flag, target = self.is_in_cache(query + str(his), data_cache=self.caches.cache)
                if flag:
                    js = target
                    return json.dumps(js, ensure_ascii=False) + '\n'
            begin = time.time()
            labels = []
            tasks = [self.process_model(model, prompt=prompt) for model in self.model_list]
            results = await asyncio.gather(*tasks)
            for result in results:
                if result:
                    for k, v in result.items():
                        if "_label" in k:
                            labels.append(v)
                            break
                    js.update(result)
            label, fre = find_most_frequent(labels)
            need_next_model = True
            if not needed_label_list:
                needed_label_list = self.need_post_pipeline_label_list
            #Only when the voting result is under the desired label will the post-processing large model be used, improving efficiency
            if needed_label_list and label not in needed_label_list:
                need_next_model = False
            #If the voting result is not the desired label, then the final label is empty.
            if not need_next_model:
                js["label_vote"] = label
                js["reason_vote"] = "voted"
                js["label"] = ""
                js["reason"] = ""
            else:
                res_temp = await self.process_model(model=self.model, prompt=prompt_post_processing)
                if res_temp:
                    js["label"] = res_temp[f"{self.model.model_name}_label"]
                    js["reason"] = res_temp[f"{self.model.model_name}_reason"]
                else:
                    js["label"] = ""
                    js["reason"] = ""
                if len(self.model_list) == 3:
                    if fre >= 2:
                        js["label_vote"] = label
                        js["reason_vote"] = "voted"
                elif len(self.model_list) >= 4:
                    if fre >= 3:
                        js["label_vote"] = label
                        js["reason_vote"] = "voted"
                else:
                    js["label_vote"] = ""
                    js["reason_vote"] = ""
                    
            output_line = json.dumps(js, ensure_ascii=False) + '\n'
            logger.info(f"single time  use :{time.time() - begin}")
            return output_line
        except Exception as e:
            logger.error(traceback.print_exc())
            return self.res_templete

    async def generate_one_by_synthsis(self):
        """
        Generate results based solely on the prompt, without reference data.
        """
        try:
            begin = time.time()
            res_temp = {}
            
            for i in range(10, 0, -1):
                result = await self.model.chat(self.prompt)
                res_temp = self.response_parser.parse_response(result)
                
                if res_temp:
                    async with self.lock:  
                        self.caches.add_cache("\n".join(list(res_temp.values())), res_temp)
                        logger.info(f"cache size:{len(self.caches.cache)}")
                    break
                    
            output_line = json.dumps(res_temp, ensure_ascii=False) + '\n'
            logger.info(f"single time  use :{time.time() - begin}")
            return output_line
            
        except Exception as e:
            logger.error(traceback.print_exc())
            return self.res_templete

    async def generate_one_by_case_rewrite(self, line: str):
        """
        Expand and rewrite based on known data and the prompt
        """
        try:
            js = json.loads(line)
            query = js.get('input')
            his = js.get('history')
            prompt = self.prompt.format(case=js)
            
            if isinstance(query, list):
                query = "\n".join(query)
                
            async with self.lock:  
                flag, target = self.is_in_cache(query+his, data_cache=self.caches.cache)
                if flag:
                    js = target
                    return json.dumps(js, ensure_ascii=False)+'\n'
                    
            begin = time.time()
            labels = []
            if "label" in js and js["label"]:
                labels.append(js["label"])
            for i in range(10, 0, -1):
                result = await self.model.chat(prompt)
                res_temp = self.response_parser.parse_response(result)
                
                if res_temp:
                    js["rewrite"] = res_temp
                    break
                    
            output_line = json.dumps(js, ensure_ascii=False)+'\n'
            logger.info(f"single time  use :{time.time()-begin}")
            return output_line
            
        except Exception as e:
            logger.error(traceback.print_exc())
            return self.res_templete

    async def generate_one_by_all_intention(self, line: str, needed_label_list: List[str] = []):
        """
        Identify the intent of queries within all content of the conversation using a prompt. 
        needed_label_list: the names of the target intent labels.
        """
        try:
            js = json.loads(line)
            query = js.get('dialogs') if "dialogs" in js else ""
            if isinstance(query, list):
                query = "\n".join(query)
                
            async with self.lock:  
                flag, target = self.is_in_cache(query, data_cache=self.caches.cache)
                if flag:
                    js = target
                    return json.dumps(js, ensure_ascii=False)+'\n'
                    
            begin = time.time()
            
            for i in range(10, 0, -1):
                result = await self.model.chat(self.prompt.format(query))
                res_temp = self.response_parser.parse_response(result)
                
                if isinstance(res_temp, dict):
                    res_temp = [res_temp]
                    
                if res_temp:
                    js["model_output"] = res_temp
                    needed = []
                    for one in res_temp:
                        if needed_label_list:
                            if one["label"] in needed_label_list:
                                needed.append(one)
                        else:
                            needed.append(one)
                    js["needed_query"] = needed
                    js["from"] = self.from_where
                    js["task"] = self.task_name
                    break
                await asyncio.sleep(1)
                
            async with self.lock:
                self.caches.add_cache(query, js)
                
            output_line = json.dumps(js, ensure_ascii=False)+'\n'
            logger.info(f"single time  use :{time.time()-begin}")
            return output_line
            
        except Exception as e:
            logger.error(traceback.print_exc())
            return self.res_templete
    
    def is_in_cache(self, query:AnyStr, data_cache:Dict[AnyStr, AnyStr]):
        """Check if the query result is already in cache"""
        reg=r'''[,.!?;:'"“”‘’\-—_(){}\[\]<>《》、，。！？；：‘’“”【】（）…]'''
        query=re.sub(reg, '', query)
        if query in data_cache and data_cache[query]:
            return True, data_cache[query]
        return False, ""

    def sampling_from_data_with_all_intentions(self, data_path: AnyStr, save_path: AnyStr, 
                                           test_file_name: Optional[AnyStr] = "test_dataset.xlsx", 
                                           train_file_name: Optional[AnyStr] = "train_dataset.json", 
                                           round: Optional[int] = 2, 
                                           one_label_total_up_threshold: int = 10000, 
                                           one_data_one_label_sampling_up_threshold: int = 4,
                                           test_data_num_threshold: Optional[int] = 100,
                                           labels: list = []) -> None:
        """
        Label each round of dialogue in multi-turn dialogue data and sample based on given (or not given) labels. 
        The main goal is to ensure diversity in sample sources.

        Parameters:
        - data_path (str): Path to the data file
        - save_path (str): Path to save the file
        - test_file_name (str): Test file name, default is "test_dataset.xlsx"
        - train_file_name (str): Training file name, default is "train_dataset.json"
        - round (int): Number of rounds, default is 2
        - one_label_total_up_threshold (int): Upper limit threshold for a single label, default is 10000
        - one_data_one_label_sampling_up_threshold (int): Upper limit threshold for sampling a single data and label, default is 4
        - labels (list): List of specified labels for sampling, default is an empty list, meaning all sample labels are sampled.

        Returns:
        None
        """
        train_test_dataset = []
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        res = {}
        error_num = 0
        num = 0
        for line in line_generator(data_path):
            num += 1
            try:
                one_dict = json.loads(line)
                dialogs = one_dict["dialogs"]
                if isinstance(dialogs, str):
                    dialogs = one_dict["dialogs"].split("\n")
                temp = DefaultDict(int)
                for intention in one_dict["model_output"]:
                    one_dict_temp = copy.deepcopy(one_dict)
                    one_dict_temp["input"] = intention["content"]
                    one_dict_temp["input"] = intention["content"]
                    one_dict_temp["history"] = "\n".join(self.find_his(intention["content"], dialogs, round=round))
                    one_dict_temp["label"] = intention["label"]
                    one_dict_temp.pop("model_output")
                    one_dict_temp.pop("needed_query")
                    one_dict_temp.pop("dialogs")
                    if temp.get(intention["label"], 0) < one_data_one_label_sampling_up_threshold:
                        res.setdefault(intention["label"], []).append(one_dict_temp)
                        temp[intention["label"]] += 1
            except Exception as e:
                error_num += 1
                logger.error(traceback.print_exc())
                continue
        random.seed(0)
        with pd.ExcelWriter(save_path + f"/{test_file_name}") as wf:
            for k, v in res.items():
                logger.info(f"(k, v):{(k, len(v))}")
                if labels and k not in labels:
                    continue
                else:
                    random.shuffle(v)
                    df = pd.DataFrame(v[:test_data_num_threshold])
                    train_test_dataset.extend(v[test_data_num_threshold:one_label_total_up_threshold])
                    df.to_excel(wf, sheet_name=k, index=False)
                with open(save_path + f"/{k}.json", "w", encoding="utf-8") as wff:
                    wff.writelines("\n".join(json.dumps(one, ensure_ascii=False) for one in v[:]))
        with open(save_path + f"/{train_file_name}", "w", encoding="utf-8") as wf:
            wf.writelines("\n".join([json.dumps(one, ensure_ascii=False) for one in train_test_dataset]))
        logger.info(f"Total data count: {num}")
        logger.info(f"Number of errors: {error_num}")
        
    def find_his(self, content: AnyStr, dialogs: List[AnyStr], round: int = 3) -> List[AnyStr]:
        """
        Find dialogue history
        
        Args:
            content: Current dialogue content
            dialogs: List of dialogues
            round: Number of historical dialogue rounds to return, default is 3
        
        Returns:
            List[AnyStr]: List of historical dialogues
        
        Raises:
            ValueError: When input parameters are invalid
        """
        # Parameter validation
        if not isinstance(dialogs, list) or not dialogs:
            return []
        if not content or not isinstance(content, (str, bytes)):
            return []
        if not isinstance(round, int) or round < 0:
            round = 2  # Use default value
        try:
            # Clean and standardize data
            cleaned_dialogs = [dialog.strip() for dialog in dialogs if dialog and isinstance(dialog, (str, bytes))]
            cleaned_content = content.strip()
            # Find the position of content in the dialogue list
            try:
                index = cleaned_dialogs.index(cleaned_content)
            except ValueError:
                return []  # Return empty list if content is not found  
            # Collect historical dialogues
            his = []
            num = 0
            # Traverse backward from the current position
            for dialog in reversed(cleaned_dialogs[:index]):
                # Check dialogue role identifier
                is_user = any(dialog.startswith(prefix) for prefix in ["用户", "患者", "User", "Patient"])                
                if is_user:
                    num += 1                    
                his.insert(0, dialog)              
                if num >= round:
                    break                   
            return his          
        except Exception as e:
            # Log error
            logger.error(f"Error in find_his: {str(e)}")
            return []

    async def generate_one_by_wizard_evolution(self, line:str)->str:
        """This method is used for instruction evolution to further evolve diversity and difficulty."""
        item = json.loads(line)
        """此方法用于指令进化，用于将多样性和难度进一步演化"""
        format_item = await convert_query_format(item, wizard_query="", task_label="", use_history=self.use_history)

        # If the data does not contain the required key field information, return directly
        if format_item["code"] == 1:
            item["wizard_round_cnt"] = -1
            item["wizard_llm_querys"] = []
            return json.dumps(item, ensure_ascii=False) + '\n'

        try:
            # Initialize the query pool
            init_query = [d["content"] for d in format_item["data"] if d["role"] == "user"]
            async with self.lock:
                flag, target = self.is_in_cache("\n".join(init_query), self.caches.cache)
                if flag:
                    return json.dumps(target, ensure_ascii=False) + '\n'
            init_task_label = format_item["task_label"]
            l_instruction_pool = [dict(instruction=init_query, task_label=init_task_label)]
            item["wizard_round_cnt"] = 0
            item["wizard_llm_querys"] = []

            for wizard_round_cnt in range(self.wizard_evolution_round):
                # Query sampling
                sample_intructions = random.sample(l_instruction_pool, min(len(l_instruction_pool), 3))
                if not sample_intructions:
                    break

                l_breadth_instructions = []
                l_depth_instructions = []

                # Breadth and depth evolution
                for idx, d in enumerate(sample_intructions):
                    #  Generate breadth and depth evolution prompts
                    breadth_prompt = await create_wizard_prompt(item=item, prompt_type="B", wizard_query=d["instruction"],
                                                       task_label=d["task_label"], use_history=self.use_history)
                    depth_prompt = await create_wizard_prompt(item=item, prompt_type="D", wizard_query=d["instruction"],
                                                     task_label=d["task_label"], use_history=self.use_history)

                    # Generate breadth evolution results
                    d_breadth_result = await self.process_model(model=self.model, prompt=breadth_prompt)
                    try:
                        if isinstance(d_breadth_result[f"{self.model.model_name}_res"], str):
                            d_breadth_result[f"{self.model.model_name}_res"] = json.loads(d_breadth_result[f"{self.model.model_name}_res"])
                    except Exception as e:
                        logger.error(str(e))
                        logger.error(f"this wizard query has format error as {d_breadth_result[self.model.model_name+'_res']}")
                        continue

                    l_breadth_result = [dict(instruction=d, task_label=item["task_label"]) for d in
                                            d_breadth_result[f"{self.model.model_name}_res"]]
          
                     # Generate depth evolution results
                    d_depth_result = await self.process_model(prompt=depth_prompt, model=self.model)
                    try:
                        if isinstance(d_depth_result[f"{self.model.model_name}_res"], str):
                            d_depth_result[f"{self.model.model_name}_res"] = json.loads(d_depth_result[f"{self.model.model_name}_res"])
                    except Exception as e:
                        logger.error(str(e))
                        logger.error(f"this wizard query has format error as {d_breadth_result[self.model.model_name+'_res']}")
                        continue
                    l_depth_result = [dict(instruction=d["input"], task_label=d["changeType"]) for d in
                                          d_depth_result[f"{self.model.model_name}_res"] if d.get("input") and d.get("changeType")]

                    # Add pre_query
                    pre_query = d["instruction"]
                    for _d in l_breadth_result:
                        _d["pre_query"] = pre_query
                    for _d in l_depth_result:
                        _d["pre_query"] = pre_query

                    # Score breadth evolution results
                    l_temp_breadth_instructions = []
                    for d in l_breadth_result:
                        breadth_eval_prompt = await create_wizard_prompt(item=item, prompt_type="C",
                                                                wizard_query=d["instruction"],
                                                                task_label=d["task_label"],
                                                                use_history=self.use_history)
                        bread_eval_result = await self.process_model(prompt=breadth_eval_prompt, model=self.model)

                        instruction = d["instruction"]
                        task_label = d["task_label"]
                        # pre_query = d["pre_query"]
                        if isinstance(bread_eval_result[f"{self.model.model_name}_res"], str):
                            bread_eval_result[f"{self.model.model_name}_res"] = json.loads(bread_eval_result[f"{self.model.model_name}_res"])

                        if bread_eval_result[f"{self.model.model_name}_res"].get("valid", 0) == 1:
                            try:
                                valid = bread_eval_result[f"{self.model.model_name}_res"].get("valid", 0)
                                score = re.search("\d+", str(bread_eval_result[f"{self.model.model_name}_res"].get("score", 0)))
                                if score and score.group():
                                    score = int(score.group())
                            except:
                                valid = -1
                                score = -999
                        else:
                            valid = bread_eval_result[f"{self.model.model_name}_res"].get("valid", -1)
                            score = bread_eval_result[f"{self.model.model_name}_res"].get("score", 0)
                        l_temp_breadth_instructions.append(
                            dict(instruction=instruction, task_label=task_label, score=score, valid=valid,
                                 pre_query=pre_query))

                    # Score depth evolution results
                    l_temp_depth_instructions = []
                    for d in l_depth_result:
                        depth_eval_prompt =await create_wizard_prompt(item=item, prompt_type="C", wizard_query=d["instruction"],
                                                              task_label=d["task_label"],
                                                              use_history=self.use_history)
                        depth_eval_result = await self.process_model(prompt=depth_eval_prompt, model=self.model)

                        instruction = d["instruction"]
                        task_label = d["task_label"]
                        # pre_query = d["pre_query"]
                        if isinstance(depth_eval_result[f"{self.model.model_name}_res"], str):
                            depth_eval_result[f"{self.model.model_name}_res"] = json.loads(depth_eval_result[f"{self.model.model_name}_res"])

                        if depth_eval_result[f"{self.model.model_name}_res"].get("valid", 0) == 1:
                            try:
                                valid = depth_eval_result[f"{self.model.model_name}_res"].get("valid", 0)
                                score = re.search("\d+", str(depth_eval_result[f"{self.model.model_name}_res"].get("score", 0)))
                                if score and score.group():
                                    score = int(score.group())
                            except:
                                valid = -1
                                score = -999
                        else:
                            valid = depth_eval_result[f"{self.model.model_name}_res"].get("valid", -1)
                            score = depth_eval_result[f"{self.model.model_name}_res"].get("score", 0)
                        l_temp_depth_instructions.append(
                            dict(instruction=instruction, task_label=task_label, score=score, valid=valid,
                                 pre_query=pre_query))

                    l_breadth_instructions.extend(l_temp_breadth_instructions)
                    l_depth_instructions.extend(l_temp_depth_instructions)

                # Save breadth and depth instruction results
                item["wizard_llm_querys"].append(dict(l_depth_instructions=l_depth_instructions,
                                                      l_breadth_instructions=l_breadth_instructions,
                                                      wizard_round_cnt=wizard_round_cnt + 1))


                # From the scored breadth and depth instructions, take the top 5% by difficulty score
                # From each round of results, select the top 20% of queries by score

                # From the breadth and depth instruction results, select the top 20% of queries
                l_instruction_pool = l_depth_instructions + l_breadth_instructions
                num_query = int(len(l_instruction_pool) * 0.2)
                l_instruction_pool = filter(lambda x: x['valid'] > 0, l_instruction_pool)
                # l_instruction_pool = filter(lambda x: x['score'] >= 50, l_instruction_pool)
                l_instruction_pool = sorted(l_instruction_pool, key=lambda x: x['score'], reverse=True)[:num_query]
            async with self.lock:
                self.caches.add_cache("\n".join(init_query), item)
        except Exception as error:
            logger.info("%" * 60)
            logger.info(f"[Instruction evolution program encountered an error! The error type is: {error}, and the error details are:\n]")
            traceback.print_exc()
        return json.dumps(item, ensure_ascii=False)+"\n"
    
    async def generate_one_by_dpo_scoring(self, line: str) -> str:
        """Asynchronous single-model generation"""
        try:
            js = json.loads(line)
            query = js.get('input')
            # Use asynchronous lock to check cache
            async with self.lock:
                flag, target = self.is_in_cache(query, self.caches.cache)
                if flag:
                    return json.dumps(target, ensure_ascii=False) + '\n'
            # Asynchronously call model
            res = await self.process_model(model=self.model, prompt=line)
            if res:
                js.update(res)
                js["from"] = self.from_where
                js["task"] = self.task_name
                async with self.lock:
                    self.caches.add_cache(query, js)
            return json.dumps(js, ensure_ascii=False) + '\n'

        except Exception as e:
            logger.error(traceback.print_exc())
            return self.res_templete
