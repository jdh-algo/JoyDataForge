# [中文](https://github.com/jdh-algo/JoyDataForge/blob/main/README.md "readme-zh")|English

# JOYDATAFORGE: Data Forging Factory

This project aims to provide an open-source toolkit for synthetic data generation tailored for large models. Through various data synthesis methods offered by this toolkit, users can efficiently construct data pipeline processes suited to different business needs, significantly enhancing data production efficiency. After data construction is completed, the toolkit also supports comprehensive evaluations of data diversity from multiple perspectives such as semantics and entities.

## Background

In the development process of large model algorithms, while the training algorithms have become relatively mature, constructing high-quality data still consumes a significant amount of human resources and time. Therefore, we need an efficient data synthesis tool to improve the production efficiency of training data for large models.

## Features

- **Data Synthesis**
  * **Single Model Synthesis**: Supports data synthesis using a single model.
  * **Multi-Model Voting Synthesis**: Generates synthetic data through voting results from multiple models.
  * **Multi-Model Joint Synthesis**: Combines the strengths of various models for data synthesis.
  * **Wizard Evolution Data Synthesis**: Uses the Wizard method for data evolution and synthesis.
  * **Magpie Model Instruction Distillation**: Conducts data distillation through Magpie model instructions.
  * **DPO Data Synthesis and Scoring**: Supports DPO data synthesis and scoring of results.
- **Data Evaluation**
  * **Minihash Filtering**: Utilizes the minihash algorithm for data filtering.
  * **Vendi-score for Semantic Diversity**: Employs vendi-score to evaluate the semantic diversity of data.
  * **Entropy and Simpson's Index for Entity Diversity**: Assesses entity diversity using entropy and Simpson's index.
- **Data Types**
  - **JSON Data**
    If data synthesis is needed from existing data, the current query should be placed in the **"input"** field, and historical content in **"history"**.
  - **Data Example**:
    ```json
    {
       "input": "After applying metronidazole, the acne on my skin showed no improvement. What should I do?",
       "target": "",
       "summary_content": "I have acne caused by mupirocin, applied metronidazole, but it had no effect.",
       "history": [],
       "tag": "Proper Medication - Dosage",
       "task_label": [
          "medicalknowledge",
          "pharmacologyknowledge"
       ],
       "num_task_label": 5
    }
    ```

## Installation

Detailed installation steps, including dependencies and configuration instructions.

```bash
#Clone the repository
git clone https://github.com/jdh-algo/JoyDataForge.git

#Enter the project directory
cd JoyDataForge

#Install dependencies
1、Create a Python environment
    1.1 It is recommended to use Anaconda to manage the virtual environment. Download it from: https://www.anaconda.com/download
    1.2 conda create -n [env_name] python=3.12
    1.3 conda activate [env_name]
2、Install Poetry:
    pip install poetry
    # If the installation fails, please refer to the official installation guide: https://python-poetry.org/docs/
3、Use Poetry to install project dependencies:
    poetry install
    #You can refer to the pyproject.toml file for specific dependencies.
```

## Usage

```bash
# Example usage of the synthesis tool - multi_model voting and post model labeling
# This example uses multiple large models to vote on the query's label, then performs secondary labeling with the specified label to enhance data quality.
# Note:
# 1. In this example, you need to implement your own model in src.joydataforge.models.llm, which can be an API or a self-deployed model.
# 2. This example specifies a prompt in src.joydataforge.config import agent_cdss_one_qa_labeling_prompt as prompt, you can implement your own prompts as needed.

import os
import sys
import asyncio
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
from loguru import logger
from src.joydataforge.models.llm import gpt4o, claude, gpt4o_mini, glm4_flash, JoyLLM
from src.joydataforge.config import agent_cdss_one_qa_labeling_prompt as prompt
from src.joydataforge.components.loader.data_load_and_process import DataLoadAndProcess
from src.joydataforge.memory.cache.data_cache import DataCache
from src.joydataforge.components.synth.joy_synth.data_generate import JoyDataGenerate

def get_args():
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--model', type=JoyLLM, default=gpt4o, help='Model used for data generation') 
    parser.add_argument('--model_list', type=list, default=[claude, gpt4o_mini, glm4_flash], help='Models for label voting') 
    parser.add_argument('--prompt', type=str, default=prompt, help='Prompt used for data generation') 
    parser.add_argument('--prompt_post_pipline', type=str, default=prompt, help='Prompt used for data generation post pipeline') 
    parser.add_argument('--data_read_path', type=str, default="datas/wizard_instruction/example.jsonl", help='Path for data reading')  
    parser.add_argument('--file_type', type=str, default="jsonl", help='Data file type')  
    parser.add_argument('--data_write_path', type=str, default="results/cdss_datas_with_votes_post_processing", help='Path to save generated data')  
    parser.add_argument('--data_write_file_name', type=str, default="cdss_agent_data_synth.json", help='File name for saving generated data')  
    parser.add_argument('--task_name', type=str, default="cdss_multi_model_vote_and_post_labeling", help='Task name')  
    parser.add_argument('--from_where', type=str, default="***", help='Data source')   
    parser.add_argument('--method', type=str, choices=[], default="vote_and_pipeline", help='Data generation method')  
    parser.add_argument('--threshold', type=int, default=10, help='Upper limit on the amount of data to process')  
    parser.add_argument('--threshold_low', type=int, default=0, help='Starting position for processing data') 
    parser.add_argument('--max_works', type=int, default=10, help='Number of parallel processing workers') 
    parser.add_argument('--data_cache_path', type=str, default="", help='Path to historical data cache (to avoid regenerating existing data). Can be empty for first-time generation') 
    parser.add_argument('--is_save_caches', type=bool, default=False, help='Whether to save generated data as historical cache') 
    parser.add_argument('--save_cache_path', type=str, default="caches/cachesjson", help='Path to save the cache data') 
    parser.add_argument('--need_post_pipeline_label_list', type=list, default=["BMI", "Expected Gestational Age", "Drug"], help='List of labels requiring multi-stage pipeline processing') 
    base_args = parser.parse_args()
    return base_args

async def main():
    args = get_args()

    data_path_destination = args.data_write_path
    if not os.path.exists(data_path_destination):
        os.makedirs(data_path_destination)
  
    # Data to be excluded or deduplicated; if there are multiple sources, multiple historical caches can be constructed
    caches_1 = DataCache()
    caches_2 = DataCache()
  
    caches_1.cache.update(caches_2.cache)
    logger.info(f"Cache size: {len(caches_1.cache)}")
  
    # Data preprocessing
    data_pre_processor = DataLoadAndProcess(path=args.data_read_path, task=args.task_name, file_type=args.file_type)
  
    # Data generation
    data_generator = JoyDataGenerate(
        model=args.model, 
        prompt=args.prompt, 
        output_file=f'{data_path_destination}/{args.data_write_file_name}_{args.task_name}.json', 
        model_list=args.model_list, 
        method=args.method, 
        caches=caches_1, 
        threshold=args.threshold, 
        threshold_low=args.threshold_low,
        prompt_post_pipline=args.prompt_post_pipline,
        need_post_pipeline_label_list=args.need_post_pipeline_label_list, 
        max_works=args.max_works,
        from_where=args.from_where, 
        task_name=args.task_name,
        data_processor=data_pre_processor)
  
    await data_generator.generate()
  
    # After processing, data can be cached or used to rebuild
    if args.is_save_caches:
        caches_1.save_cache(save_path=args.save_cache_path)
    logger.info(f"Dataset: {args.data_read_path}, labeling processing completed!")

if __name__ == '__main__':
    asyncio.run(main())

```

Run the script with:

```bash
python examples/model_votes/model_votes_and_pipline.py
```

For more examples, refer to the contents in the [examples/ ](https://github.com/jdh-algo/JoyDataForge/tree/main/examples "examples")directory.

## **Datasets**

We synthesis two datasets by using the [Magpie](https://arxiv.org/abs/2406.08464 "喜鹊") method and [Wizard](https://arxiv.org/abs/2304.12244 "WizardLM") method of [JoyDataForge](https://github.com/jdh-algo/JoyDataForge "数据合成 锻造厂") ，And it has been open-sourced for everyone's reference and use.

Dataset 1:[joy_common_sft_datasets](https://huggingface.co/datasets/jdh-algo-huggingface/joy_common_sft_datasets "magpie"), this dataset is distilled using the magpie method on the [Qwen2-7b-Intruction](https://github.com/QwenLM/Qwen) model. It can be used for the **initial** SFT instruction fine-tuning phase of the model.

Dataset 2:[query_diversity_evolution](https://huggingface.co/datasets/jdh-algo-huggingface/query_diversity_evolution "WizardLM"), this dataset uses the wizard method to evolve the query content in the given dataset in terms of depth and breadth over two rounds. We have also scored the difficulty of tasks for the evolved queries to enhance the diversity of queries and the complexity of tasks. This is used for the **later stage** of SFT instruction fine-tuning of the model.

## Contribution

Instructions on how to contribute to the project, including reporting bugs, suggesting features, or submitting code.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## Citation

```plaintext
@misc{joydataforge2024,
  title={JoyDataForge: An Open Source Toolkit for Large-Scale Data Synthesis},
  author={weili36190429 and lb2158},
  year={2024},
  howpublished={\url{https://github.com/jdh-algo/JoyDataForge.git}},
  note={Version 1.0}
}
```

## Contributors


<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-13-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->


<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/weili36190429"><img src="https://avatars.githubusercontent.com/weili36190429?s=100" width="100px;" alt="User's Name"/><br /><sub><b>User's Name</b></sub></a><br /><a href="https://github.com/jdh-algo/JoyDataForge/jdh-algo/JoyDataForge/commits?author=weili36190429" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lb2158"><img src="https://avatars.githubusercontent.com/lb2158?s=100" width="100px;" alt="User's Name"/><br /><sub><b>User's Name</b></sub></a><br /><a href="https://github.com/jdh-algo/JoyDataForge/jdh-algo/JoyDataForge/pulls?q=is%3Apr+reviewed-by%3Alb2158" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ray075hl"><img src="https://avatars.githubusercontent.com/ray075hl?s=100" width="100px;" alt="User's Name"/><br /><sub><b>User's Name</b></sub></a><br /><a href="#tool-ray075hl" title="Tools">🔧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/helizhi1"><img src="https://avatars.githubusercontent.com/helizhi1?s=100" width="100px;" alt="User's Name"/><br /><sub><b>User's Name</b></sub></a><br /><a href="#tool-helizhi1" title="Tools">🔧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ysleep"><img src="https://avatars.githubusercontent.com/ysleep?s=100" width="100px;" alt="User's Name"/><br /><sub><b>User's Name</b></sub></a><br /><a href="#ideas-ysleep" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Luca202412"><img src="https://avatars.githubusercontent.com/Luca202412?s=100" width="100px;" alt="User's Name"/><br /><sub><b>User's Name</b></sub></a><br /><a href="#ideas-Luca202412" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/beiyouwuyanzu"><img src="https://avatars.githubusercontent.com/beiyouwuyanzu?s=100" width="100px;" alt="User's Name"/><br /><sub><b>User's Name</b></sub></a><br /><a href="#example-beiyouwuyanzu" title="Examples">💡</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Miss_wanyueli"><img src="https://avatars.githubusercontent.com/Miss_wanyueli?s=100" width="100px;" alt="User's Name"/><br /><sub><b>User's Name</b></sub></a><br /><a href="#example-Miss_wanyueli" title="Examples">💡</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/stalwart0465"><img src="https://avatars.githubusercontent.com/stalwart0465?s=100" width="100px;" alt="User's Name"/><br /><sub><b>User's Name</b></sub></a><br /><a href="#content-stalwart0465" title="Content">🖋</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/minyu.gao@outlook.com"><img src="https://avatars.githubusercontent.com/minyu.gao@outlook.com?s=100" width="100px;" alt="User's Name"/><br /><sub><b>User's Name</b></sub></a><br /><a href="#content-minyu.gao@outlook.com" title="Content">🖋</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/haishenmuc@163.com"><img src="https://avatars.githubusercontent.com/haishenmuc@163.com?s=100" width="100px;" alt="User's Name"/><br /><sub><b>User's Name</b></sub></a><br /><a href="#content-haishenmuc@163.com" title="Content">🖋</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/JackZhang199"><img src="https://avatars.githubusercontent.com/JackZhang199?s=100" width="100px;" alt="User's Name"/><br /><sub><b>User's Name</b></sub></a><br /><a href="#content-JackZhang199" title="Content">🖋</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kkbossa"><img src="https://avatars.githubusercontent.com/kkbossa?s=100" width="100px;" alt="User's Name"/><br /><sub><b>User's Name</b></sub></a><br /><a href="#content-kkbossa" title="Content">🖋</a></td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td align="center" size="13px" colspan="7">
        <img src="https://raw.githubusercontent.com/all-contributors/all-contributors-cli/1b8533af435da9854653492b1327a23a4dbd0a10/assets/logo-small.svg">
          <a href="https://all-contributors.js.org/docs/en/bot/usage">Add your contributions</a>
        </img>
      </td>
    </tr>
  </tfoot>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->



## Contact Information

* Email: [liwei1559@jd.com](mailto:liwei1559@jd.com)
