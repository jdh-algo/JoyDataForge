"""
This module defines the `MagpieGenerator` class, which is responsible for generating user instructions and corresponding model responses using a local language model.

The file includes the following import statements:
- os
- argparse
- json
- random
- numpy as np
- asyncio
- torch
- tqdm
- Any, List, Dict from typing
- SamplingParams from vllm
- BaseModel, Field from pydantic
- AutoTokenizer from transformers
- instruction_post_process from src.joydataforge.utils.magpie_utils.str_utils
- logger from loguru
- LocalLLM from src.joydataforge.models.llm

The file also includes the following classes and functions:
- MagpieGenerator
  - __init__
  - setup_args
  - set_random_seed
  - setup_output
  - setup_device
  - setup_model
  - load_config
  - setup_sampling_params_instrcutions
  - setup_sampling_params_responses
  - de_md_logits_processor_for_llama3_1
  - generate_batch_instructions
  - generate_batch_responses
  - create_result_dict
  - generate
  - generate_multi_conversation
  - save_results

To use this module, you can instantiate the `MagpieGenerator` class with appropriate arguments and call its `generate` method to create instructions and responses.
"""

import os
import argparse
import json
import random
import numpy as np
import asyncio
import torch
from tqdm import tqdm
from typing import Any, List, Dict
from vllm import SamplingParams
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from src.joydataforge.utils.magpie_utils.str_utils import instruction_post_process
from loguru import logger
from src.joydataforge.models.llm import LocalLLM


class MagpieGenerator(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }
    args: argparse.Namespace = Field(...)
    llm: Any = Field(None)
    tokenizer: Any = Field(None)
    pre_query_template: str = Field(None)
    stop_tokens: List[str] = Field(default_factory=list)
    stop_token_ids: List[int] = Field(default_factory=list)
    ins_sampling_params: Any = Field(None)
    res_sampling_params: Any = Field(None)
    output_dir: str = Field(None)

    def __init__(self, args: argparse.Namespace, **kwargs):
        super().__init__(args=args, **kwargs)
        self.args = args
        self.setup_args()
        self.setup_output()
        self.setup_device()
        self.setup_model()
        self.load_config()
        self.setup_sampling_params_instrcutions()
        self.setup_sampling_params_responses()
        self.results = []

    def setup_args(self):
        if self.args.total_prompts is None:
            if self.args.repeat is None:
                raise ValueError("Either total prompts or repeat should be specified.")
            self.args.total_prompts = self.args.repeat * self.args.n
        else:
            self.args.repeat = int(np.ceil(self.args.total_prompts / self.args.n))

        if self.args.seed is not None:
            self.set_random_seed()

    def set_random_seed(self):
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)

    def setup_output(self):
        res = "_and_res" if self.args.generate_response_immediately else ""
        output_filename = f"Magpie_{self.args.model.split('/')[-1]}_{self.args.total_prompts}_{self.args.timestamp}_ins{res}.json"
        if not self.args.job_name:
            if not os.path.exists(self.args.output_folder):
                os.makedirs(self.args.output_folder)
            self.output_dir = f"{self.args.output_folder}/{output_filename}"
        else:
            self.output_dir = f"{self.args.output_folder}/{self.args.job_name}/{output_filename}"

    def setup_device(self):
        if self.args.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.device

    def setup_model(self):
        logger.info(self.args)
        self.llm = LocalLLM(
            model=self.args.model,
            dtype=self.args.dtype,
            gpu_memory_utilization=self.args.gpu_memory_utilization,
            max_model_len=self.args.max_model_len,
            tensor_parallel_size=self.args.tensor_parallel_size
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)

    def load_config(self):
        with open(self.args.magpie_model_templetes_path, "r", encoding="utf-8") as f:
            model_configs = json.load(f)
            model_config = model_configs[self.args.model]

            if self.args.control_tasks:
                self.pre_query_template = model_config[f"pre_query_template_{self.args.control_tasks}"]
            elif self.args.system_prompt:
                self.pre_query_template = model_config["pre_query_template_with_system_prompt"]
            else:
                self.pre_query_template = model_config["pre_query_template"]

            self.stop_tokens = model_config["stop_tokens"] + model_config["stop_tokens_assistant"]
            self.stop_token_ids = model_config["stop_token_ids"]

            if self.args.early_stopping:
                self.stop_tokens.append("\n")

    def setup_sampling_params_instrcutions(self):
        logits_processor = None
        if self.args.logits_processor and "llama-3.1" in self.args.model.lower():
            logits_processor = self.de_md_logits_processor_for_llama3_1
        self.ins_sampling_params = SamplingParams(
            n=self.args.n,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            max_tokens=self.args.max_tokens,
            skip_special_tokens=self.args.skip_special_tokens,
            stop=self.stop_tokens,
            stop_token_ids=self.stop_token_ids,
            logits_processors=[logits_processor] if logits_processor else None
        )

    def setup_sampling_params_responses(self):
        self.res_sampling_params = SamplingParams(
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            max_tokens=self.args.max_tokens,
            skip_special_tokens=self.args.skip_special_tokens,
            stop=self.stop_tokens,
            stop_token_ids=self.stop_token_ids,
            repetition_penalty=self.args.repetition_penalty
        )

    @staticmethod
    def de_md_logits_processor_for_llama3_1(token_ids, logits):
        if len(token_ids) == 0:
            logits[2] = -9999.999
            logits[567] = -9999.999
            logits[14711] = -9999.999
            logits[827] = -9999.999
        return logits

    async def generate_batch_instructions(self, round_num: int) -> List[Dict]:
        """Generate a batch of user instruction data """
        batch_results = []
        output = await asyncio.to_thread(self.llm.generate, self.pre_query_template, self.ins_sampling_params)
        output_list = output[0].outputs
        if self.args.shuffle:
            random.shuffle(output_list)
        for i, completion in enumerate(output_list):
            instruction = completion.text.strip() if self.args.engine == "vllm" else completion.strip()
            result = await self.create_result_dict(round_num * self.args.n + i, instruction)
            batch_results.append(result)
        return batch_results

    async def generate_batch_responses(self, batch):
        """Generate model responses corresponding to user instructions"""
        user_instructions = [item['sanitized_instruction'] if self.args.sanitize else item['instruction'] for item in batch]
        prompts = []
        for instruction in user_instructions:
            chat = [{"role": "user", "content": instruction}]
            template = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            prompts.append(template)
        outputs = self.llm.generate(prompts, self.res_sampling_params)
        for i, item in enumerate(batch):
            item['response'] = outputs[i].outputs[0].text.strip()
            item['gen_response_configs'] = {
                "prompt": prompts[i],
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "repetition_penalty": self.args.repetition_penalty,
                "max_tokens": self.args.max_tokens,
                "stop_tokens": self.stop_tokens,
                "output_generator": self.args.model,
                "engine": self.args.engine,
            }
        return batch

    async def create_result_dict(self, id: int, instruction: str) -> Dict:
        if self.args.sanitize:
            sanitized_instruction, class_num = await instruction_post_process(instruction, self.args.model)
            return {
                "id": id,
                "pre_query_template": self.pre_query_template,
                "instruction": instruction,
                "sanitized_instruction": sanitized_instruction,
                "class": class_num
            }
        else:
            return {
                "id": id,
                "pre_query_template": self.pre_query_template,
                "instruction": instruction
            }

    async def generate(self):
        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)
        with open(self.output_dir, 'w', encoding='utf-8') as wf:
            for round_num in tqdm(range(self.args.repeat)):
                batch_results = await self.generate_batch_instructions(round_num)
                if self.args.generate_response_immediately:
                    batch_results = await self.generate_batch_responses(batch_results)
                    for one in batch_results:
                        wf.write(json.dumps(one, ensure_ascii=False) + "\n")
                    wf.flush()
                self.results.extend(batch_results)
        return self.results
