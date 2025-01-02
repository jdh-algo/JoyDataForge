"""
This module defines several classes for interacting with language models, both locally and via APIs.

The file includes the following import statements:
- traceback
- aiohttp
- asyncio
- base64
- struct
- json
- numpy as np
- torch
- LLM, SamplingParams from vllm
- AsyncOpenAI from openai
- AnyStr, Dict, Any, Union, Optional, List from typing
- logger from loguru
- base_model_app_key, base_model_url_outside from src.joydataforge.config
- embedding_model_name, embedding_model_url from src.joydataforge.config
- AutoTokenizer, AutoModelForSequenceClassification from transformers

The file also includes the following classes and functions:
- EmbeddingModel
  - __init__
  - get_single_embedding
  - get_all_embeddings
  - close
- LocalLLM
  - __init__
- LocalLLMHF
  - __init__
- JoyLLM
  - __init__
  - chat
  - get_chatrhino_res
  - get_claude_res
  - get_gemini_res
  - get_openai_gpt_http_res
  - get_openai_gpt_sdk_res
  - get_zhupu_res
  - get_qwen_res
  - get_kimi_res
  - get_perplexity_res
  - get_local_reward_model_res

To use this module, you can instantiate the appropriate class for your use case, such as `EmbeddingModel` for obtaining text embeddings, `LocalLLM` for local model deployment, or `JoyLLM` for interacting with models via API calls.
"""

import traceback
import aiohttp
import asyncio
import base64
import struct
import json
import numpy as np
import torch
from vllm import LLM, SamplingParams
from openai import AsyncOpenAI
from typing import AnyStr, Dict, Any, Union, Optional, List
from loguru import logger
from src.joydataforge.config import base_model_app_key, base_model_url_outside
from src.joydataforge.config import embedding_model_name, embedding_model_url
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Vectorization
class EmbeddingModel():
    """Handles text embedding operations."""
    
    def __init__(self):
        self.embedding_model_name = embedding_model_name
        self.embedding_model_url = embedding_model_url
        
    async def get_single_embedding(self, text: str, is_query: int = 1) -> List[float]:
        """
        Asynchronously retrieves the embedding for a given text.

        Args:
            text: The input text.
            is_query: Indicates if the text is a query.

        Returns:
            A list representing the embedding vector.
        """
        try:
            encoded_text = base64.b64encode(text.encode('utf-8')).decode('utf-8')
            encoded_is_query = base64.b64encode(str(is_query).encode('utf-8')).decode('utf-8')

            payload = {
                "model_name": self.embedding_model_name,
                "inputs": [
                    {"name": "texts", "datatype": "BYTES", "shape": [1],
                    "contents": {"bytes_contents": [encoded_text]}},
                    {"name": "is_query", "datatype": "BYTES", "shape": [1],
                    "contents": {"bytes_contents": [encoded_is_query]}}
                ],
                "outputs": [{"name": "text_emb"}]
            }

            async with self.session as session:
                async with session.post(
                    self.embedding_model_url,
                    json=payload
                ) as response:
                    result = await response.json()
                    decoded = base64.b64decode(result['raw_output_contents'][0])
                    num_floats = len(decoded) // 4
                    return struct.unpack('<' + 'f' * num_floats, decoded)
        except Exception as e:
            logger.error(f"Error occurred while getting embedding for text '{text}': {e}")
            return []

    async def get_all_embeddings(self, datas: List[Dict[str, Any]]) -> np.ndarray:
        """
        Initializes the embedding features for a dataset.

        Args:
            datas: A list of data containing query texts.

        Returns:
            np.ndarray: An embedding matrix.

        Raises:
            Exception: Errors during the embedding retrieval process.
        """
        if datas:
            self.datas = datas

        try:
            # Extract query texts
            queries = [
                json.loads(item)["input"] if isinstance(item, str) else item["input"]
                for item in self.datas
            ]

            # Create session if not exists
            if not hasattr(self, 'session') or self.session.closed:
                self.session = aiohttp.ClientSession()

            # Create tasks for all embedding requests
            tasks = [self.get_single_embedding(query) for query in queries]
            
            # Execute all tasks concurrently
            embeddings = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for errors
            for emb in embeddings:
                if isinstance(emb, Exception):
                    raise emb

            return np.array(embeddings)

        except Exception as e:
            logger.error(f"Error occurred while getting embeddings: {str(e)}")
            return np.array([])

        finally:
            # Ensure session is properly closed
            if hasattr(self, 'session') and not self.session.closed:
                await self.session.close()
                
    async def close(self):
        """Method to close resources."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

# Local deployment model -- vllm deployment
class LocalLLM(LLM):
    """Locally deployed large language model.

    Attributes:
        model (str): Model path
        dtype (str): Data type
        max_model_len (int): Maximum model length
        gpu_memory_utilization (float): GPU memory utilization rate
        trust_remote_code (bool): Whether to trust remote code
        tensor_parallel_size (int): Tensor parallel size
        seed (int): Random seed
        swap_space (int): Swap space size
    """

    def __init__(
        self,
        model: str,
        dtype: str = "float16",
        max_model_len: int = 30000,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        tensor_parallel_size: int = 2,
        seed: int = 0,
        swap_space: int = 4
    ) -> None:
        """Initializes LocalLLM.

        Args:
            model (str): Model path
            dtype (str): Data type
            max_model_len (int, optional): Maximum model length. Defaults to 4096.
            gpu_memory_utilization (float, optional): GPU memory utilization rate. Defaults to 0.9.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to True.
            tensor_parallel_size (int, optional): Tensor parallel size. Defaults to 2.
            seed (int, optional): Random seed. Defaults to 0.
            swap_space (int, optional): Swap space size. Defaults to 4.

        Raises:
            ValueError: Raised when parameter values are invalid.
        """
        super().__init__(
            model=model,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            swap_space=swap_space
        )
        
        # 参数验证
        if not isinstance(model, str) or not model:
            raise ValueError("model_path must be a non-empty string")
        
        if not 0 < gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
        
        if max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        
        if tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        
        if swap_space < 0:
            raise ValueError("swap_space must be non-negative")

        # 属性赋值
        self.model = model
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.tensor_parallel_size = tensor_parallel_size
        self.seed = seed
        self.swap_space = swap_space
        
##deploy local model--huggingface#
class LocalLLMHF():
    """Locally deployed large language model.
    """

    def __init__(
        self,
        model: str,
        trust_remote_code: bool = True,
        dtype = torch.bfloat16,
        device_map:str = "auto"
   
    ) -> None:
        """Initializes LocalLLM.
        Args:
            model (str): Model path
            dtype (str): Data type
            device_map (str): Device Map
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to True

        Raises:
            ValueError: Raised when parameter values are invalid.
        """
        # Parameter validation
        if not isinstance(model, str) or not model:
            raise ValueError("model_path must be a non-empty string")

        # Attribute assignment
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype, 
            device_map=device_map
        )

 
class JoyLLM:
    """Handles interactions with language models via API calls.
    """
    def __init__(self, model:Optional[AnyStr]="", url:Optional[AnyStr]="", app_key:Optional[AnyStr]="", is_local_rm_mode:bool=False, engine:str="vllm", args:Optional[AnyStr]="", **kwargs) -> None:
        self.args= args
        self.model_name = model if model else self.args.model.split("/")[-1]
        self.url = url
        self.app_key = app_key
        self.engine = engine
        self.is_local_rm_mode=is_local_rm_mode
        self.local_model=None
        if self.is_local_rm_mode:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)
            if engine == "vllm":
                self.local_model=LocalLLM(
                    model=self.args.model,
                    dtype=self.args.dtype,
                    gpu_memory_utilization=self.args.gpu_memory_utilization,
                    max_model_len=self.args.max_model_len,
                    tensor_parallel_size=self.args.tensor_parallel_size
                )
            elif engine == "hf":
                self.local_model=LocalLLMHF(
                    model=self.args.model
                ).model

    async def chat(self, query:AnyStr, temperature:Optional[float]=0.3) -> Union[Dict[str, Any], AnyStr]:
        try:
            if 'Qwen' in self.model_name:
                logger.info(self.model_name, "Model Call")
                return await self.get_qwen_res(query=query, model=self.model_name, url=self.url, app_key=self.app_key)
            elif "Chatrhino" in self.model_name:
                logger.info(self.model_name, "Model Call")
                return await self.get_chatrhino_res(query=query, model=self.model_name, url=self.url, app_key=self.app_key)
            elif "glm" in self.model_name:
                logger.info(self.model_name, "Model Call")
                return await self.get_zhupu_res(query=query, model=self.model_name, url=self.url, app_key=self.app_key)
            elif 'moonshot' in self.model_name:
                return await self.get_kimi_res(query=query, model=self.model_name, url=self.url, app_key=self.app_key,temperature=0.3)
            elif "perplexity" in self.model_name:
                logger.info(self.model_name, "Model Call")
                return await self.get_perplexity_res(query=query, model=self.model_name, url=self.url, app_key=self.app_key,temperature=0.2)
            elif "claude" in self.model_name:
                logger.info(self.model_name, "Model Call")
                return await self.get_claude_res(query=query, model=self.model_name, url=self.url, app_key=self.app_key,temperature=0.2)
            elif "gemini" in self.model_name:
                logger.info(self.model_name, "Model Call")
                return await self.get_gemini_res(query=query, model=self.model_name, url=self.url, app_key=self.app_key, temperature=0.2)
            elif self.is_local_rm_mode:
                logger.info(self.model_name, "Model Call")
                return await self.get_local_reward_model_res(query=query, is_for_dpo_scoring=True)
            else:
                logger.info(self.model_name, "Model Call")
                return await self.get_openai_gpt_http_res(query=query, model=self.model_name, url=self.url, app_key=self.app_key)
        except:
            logger.error("Model Call failed ！")
            logger.error(traceback.print_exc())
            logger.info("text:{query}")
    
    @staticmethod   
    async def get_chatrhino_res(query:AnyStr, model:AnyStr, url:AnyStr, app_key:AnyStr, temperature=0.8)-> Union[AnyStr, Dict[AnyStr, Any]]:
        headers = {
            "Authorization": f"Bearer {app_key}",
            "Content-Type": "application/json",
            "charset": "utf-8"
        }

        data = {
            "messages": [
                {
                    "content": query,
                    "role": "user"
                }
            ],
            "stream": False,
            "model": model
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, headers=headers, timeout=300) as resp:
                    if resp.status == 200:
                        ret = await resp.text()
                        return ret
        except Exception as err:
            logger.error(f"Error occurred: {str(err)}")
            return ""

    @staticmethod 
    async def get_claude_res(query:AnyStr, model:AnyStr, url:AnyStr, app_key:AnyStr, temperature=0.8)-> Union[AnyStr, Dict[AnyStr, Any]]:
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "temperature": temperature,
            "max_tokens": 4096
        }
        header = {
            "Content-Type": "application/json;charset=UTF-8",
            "Authorization": f"Bearer {app_key}"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, headers=header) as resp:
                    if resp.status == 200:
                        return await resp.text()
        except Exception as err:
            logger.error(f"Error occurred: {str(err)}")
            return ""

    @staticmethod
    async def get_gemini_res(query: AnyStr, model: AnyStr, url: AnyStr, app_key: AnyStr, temperature=0.8) -> Union[AnyStr, Dict[AnyStr, Any]]:
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "temperature": temperature,
            "max_tokens": 4096
        }
        
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "Authorization": f"Bearer {app_key}"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return await response.text()                  
        except Exception as err:
            logger.error(f"Error occurred: {str(err)}")
            return ""

    @staticmethod
    async def get_openai_gpt_http_res(query: AnyStr, model: AnyStr, url: AnyStr, app_key: AnyStr, temperature=0.8) -> Union[AnyStr, Dict[AnyStr, Any]]:
        data = {
            "app_key": app_key,
            "method": "jdh.llm.chat",
            "messages": [
                {
                    "role": "system",
                    "content": "\nYou are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024/06/05 11:09:20\nCurrent model: gpt-4-1106-preview\nCurrent time: 2024/06/05 11:09:20"
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "model": model,
            "stream": False
        }
        
        headers = {
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Host': 'llm.jdh.com',
            'Connection': 'keep-alive',
            "Authorization": f"Bearer {app_key}",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=600)  # 保持原有的600秒超时设置
                ) as response:
                    return await response.text()
                    
        except asyncio.TimeoutError:
            logger.error("timeout error!")
            return ""
        except Exception as err:
            logger.error(f"Error occurred: {str(err)}")
            return ""

    @staticmethod 
    async def get_openai_gpt_sdk_res(query:AnyStr, model:AnyStr, url:AnyStr, app_key:AnyStr, temperature=0.8)-> Union[AnyStr, Dict[AnyStr, Any]]:
        client = AsyncOpenAI(
            app_key=app_key,
            base_url=url
        )
        
        try:
            response = await client.chat.completions.create(
                model=model if model else "gpt-4-1106-preview",
                messages=[{"content": query, "role": "user"}],
                temperature=temperature,
                max_tokens=1280,
                top_p=0.0000001,
                frequency_penalty=0,
                presence_penalty=0,
                stream=False,
                timeout=120
            )
            return response.choices[0].message.content
        except Exception as err:
            logger.error(f"Error occurred: {str(err)}")
            return ""

    @staticmethod 
    async def get_zhupu_res(query: AnyStr, model: AnyStr, url: AnyStr, app_key: AnyStr, temperature=0.9) -> Union[AnyStr, Dict[AnyStr, Any]]:
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "model": model,
            "stream": False
        }
        
        headers = {
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Host': 'llm.jdh.com',
            'Connection': 'keep-alive',
            "Authorization": f"Bearer {app_key}",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=600) 
                ) as response:
                    return await response.text()
                    
        except asyncio.TimeoutError:
            logger.error("timeout error!")
            return ""
        except Exception as err:
            logger.error(f"Error occurred: {str(err)}")
            return ""

    @staticmethod 
    async def get_qwen_res(query:AnyStr, model:AnyStr, url:Optional[AnyStr]="", app_key:Optional[AnyStr]="", temperature=0.8) -> Union[AnyStr, Dict[AnyStr, Any]]:
        client = AsyncOpenAI(
            app_key=app_key,
            base_url=url if url else "http://localhost:8008/v1",
        )
        
        try:
            response = await client.chat.completions.create(
                model=model if model else "Qwen2-72B-Instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query}
                ],
                stream=False,
                max_tokens=10000,
                timeout=600,
                temperature=temperature,
                stop=['<|im_end|>', '<|endoftext|>']
            )
            return response.choices[0].message.content
        except Exception as err:
            logger.error(f"Error occurred: {str(err)}")
            return ""

    @staticmethod 
    async def get_kimi_res(query:AnyStr, model:AnyStr, url:AnyStr, app_key:AnyStr, temperature=0.3) -> Union[AnyStr, Dict[AnyStr, Any]]:
        try:
            client = AsyncOpenAI(
                app_key=app_key,
                base_url=url,
            )

            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
                    {"role": "user", "content": query}
                ],
                temperature=temperature,
            )
            result = completion.choices[0].message.content
            return result
        except Exception as err:
            logger.error(f"Error occurred: {str(err)}")
            return ""

    @staticmethod 
    async def get_perplexity_res(query: AnyStr, model: AnyStr, url: AnyStr, app_key: AnyStr, temperature=0.2) -> Union[AnyStr, Dict[AnyStr, Any]]:
        data = {
            "model": model if model else "llama-3-sonar-small-32k-online",
            "messages": [
                {
                    "content": query,
                    "role": "system"
                }
            ],
            "max_tokens": 0,
            "temperature": 0.2,
            "top_p": 0.9,
            "return_citations": False,
            "return_images": False,
            "top_k": 0,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1
        }
        
        headers = {
            "accept": "application/json",
            "content-type": "application/json"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url if url else "https://api.perplexity.ai/chat/completions",
                    json=data,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        logger.info(f"Model Call failed! error code:{response.status}")
                        return {}
                    
                    response_json = await response.json()
                    return response_json['choices'][0]['message']['content']
                    
        except Exception as err:
            logger.error(f"Error occurred: {str(err)}")
            return ""

    async def get_local_reward_model_res(self, query: Union[AnyStr, List[AnyStr]], is_for_dpo_scoring=False) -> Union[AnyStr, Dict[AnyStr, Any]]:
        """
        query: 1. User's question, used to request the model's answer
            2. Conversation information, user query, and multiple candidate responses from the model, used for the reward model to score the generated multiple candidate answers
        """


        def prepare_dpo_reward_input(query):
            prepare_res = []
            if is_for_dpo_scoring:
                if isinstance(query, str):
                    query = json.loads(query)
                user_query = query.get("input", "")
                response_candidates = query.get("response_candidates", [])
                if not user_query or (not response_candidates or not isinstance(response_candidates, list)):
                    logger.error(f"Current data has issues, missing user question or model responses!\n{query}")
                    return prepare_res

                for candidate in response_candidates:
                    res = []
                    res.append({"role": "user", "content": user_query})
                    res.append({"role": "assistant", "content": candidate})
                    prepare_res.append(res)
                return prepare_res
            else:
                prepare_res.append([{"role": "user", "content": query}])
                return prepare_res

        try:
            chat_messages = prepare_dpo_reward_input(query)
            scores = []
            for one_chat_message in chat_messages:
                # As of 2024-12-15, vllm does not support ArmoRM-Llama3-8B-v0.1 deployment
                if self.engine == "vllm":
                    res_sampling_params = SamplingParams(
                        temperature=self.args.temperature,
                        top_p=self.args.top_p,
                        max_tokens=self.args.max_tokens,
                        skip_special_tokens=self.args.skip_special_tokens,
                        stop=self.stop_tokens,
                        stop_token_ids=self.stop_token_ids,
                        repetition_penalty=self.args.repetition_penalty
                    )
                    template = self.tokenizer.apply_chat_template(one_chat_message,
                                                                tokenize=False,
                                                                add_generation_prompt=True
                                                            )
                    output = self.local_model.generate(template, res_sampling_params)
                    scores.append(output.outputs[0].text.strip())

                elif self.engine == "hf":
                    device = self.local_model.device
                    template = self.tokenizer.apply_chat_template(one_chat_message,
                                                                return_tensors="pt",
                                                                padding=True,
                                                                truncation=True,
                                                                max_length=self.args.max_model_len
                                                            ).to(device)
                    # template = self.tokenizer.apply_chat_template(chat_messages, 
                    #                                             tokenize=False, 
                    #                                             add_generation_prompt=True
                    #                                         ) + self.tokenizer.eos_token
                    # input_ids = self.tokenizer(template, return_tensors="pt").to(device)

                    with torch.no_grad():
                        output = self.local_model(template)
                        score = output.score.float().item()
                        scores.append(score)
            return {"reward_model_scores": scores}

        except Exception as e:
            logger.error(str(e))
            traceback.print_exc()
            return []


glm4 = JoyLLM(model='glm-4-plus', app_key=base_model_app_key, url=base_model_url_outside)
glm4_flash = JoyLLM(model='glm-4-flash', app_key=base_model_app_key, url=base_model_url_outside)
moonshot = JoyLLM(model='moonshot-v1-8k' , app_key=base_model_app_key, url=base_model_url_outside)
chatrhino = JoyLLM(model='Chatrhino-81B', app_key=base_model_app_key, url=base_model_url_outside)
gpt4o = JoyLLM(model='gpt-4o-0806', app_key=base_model_app_key, url=base_model_url_outside)
gpt4o_mini = JoyLLM(model='gpt-4o-mini', app_key=base_model_app_key, url=base_model_url_outside)
claude = JoyLLM(model="anthropic.claude-3-5-sonnet-20241022-v2:0", app_key=base_model_app_key, url=base_model_url_outside)
gemini = JoyLLM(model="gemini-1.0-pro-001", app_key=base_model_app_key, url=base_model_url_outside)

