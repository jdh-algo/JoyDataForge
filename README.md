## 中文|[English](https://github.com/jdh-algo/JoyDataForge/blob/main/README-EN.md "readme-en")

# HJOYDATAFORGE:数据锻造工厂

本项目旨在提供一个用于大模型数据合成的开源工具箱。通过该工具箱提供的多种数据合成方法，用户可以高效地构建适用于不同业务需求的数据管道流程，从而显著提高数据生产效率。在数据构建完成后，工具箱还支持从语义和实体等多个层面对数据的多样性进行全面评估。

## 背景介绍

在大模型算法的开发过程中，虽然模型训练算法已经相对成熟，但高质量数据的构建仍然耗费大量的人力和时间。因此，我们需要一个高效的数据合成工具来提高大模型训练数据的生产效率。

## 功能特点

- **数据合成**
  * **单模型合成** ：支持单一模型的数据合成。
  * **多模型投票合成** ：通过多个模型的结果投票生成合成数据。
  * **多模型级联合成** ：结合多种模型的优势进行数据合成。
  * **Wizrd演化数据合成** ：使用Wizrd方法进行数据演化和合成。
  * **Magpie模型指令数据蒸馏** ：通过Magpie模型指令进行数据蒸馏。
  * **DPO数据合成及打分** ：支持DPO数据合成并对结果进行评分。
- **数据评估**
  * **Minihash过滤** ：利用minihash算法进行数据过滤。
  * **Vendi-score评估语义多样性** ：使用vendi-score评估数据的语义多样性。
  * **信息熵和辛普森指数评估实体多样性** ：通过信息熵和辛普森指数进行实体多样性评估。
- 数据类型
  - **json数据**
    如果需要从已有数据进行合成， 当前query放在**”input"**字段， 历史内容放在**“history"**中。
  - **数据样例**：
    ```
    {
       "input": "涂了甲硝唑后皮肤长的痘痘没有效果，我应该如何处理？",
       "target": "",
       "summary_content": "我皮肤长了莫匹罗星的痘痘,我涂了甲硝唑,但是没有效果。",
       "history": [],
       "tag": "合理用药-用法用量",
       "task_label": [
          "medicalknowledge",
          "pharmacologyknowledge"
       ],
       "num_task_label": 5
    }
    ```

## 安装

```bash
#代码下载
git clone https://github.com/jdh-algo/JoyDataForge.git

#进入工程路径
cd JoyDataForge

#安装依赖
1、python 环境创建
    1.1 推荐 Anaconda 来管理虚拟环境， 下载地址: https://www.anaconda.com/download
    1.2 conda create -n [env_name] python=3.12
    1.3 conda activate [env_name]
2、安装 Poetry:
    pip install poetry
    # 如果安装失败，请根据官网推荐的安装方式: https://python-poetry.org/docs/
3、用 Poetry 安装工程依赖:
    poetry install
    #参考文件pyproject.toml中依赖及版本要求.

```

## 使用说明

```bash
# 合成工具的使用示例-multi_model voting and post model labeling
# 本示例使用多个大模型投票获取query的标签， 然后指定标签的数据进行二次打标签提升数据的质量
#注意：
 1、本样例是在需要在src.joydataforge.models.llm中实现自己的模型，可以是api、可以是自己部署模型；
 2、本样例在src.joydataforge.config import agent_cdss_one_qa_labeling_prompt as prompt 指定了提示词， 你可以根据自己的需求实现自己的提示词

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
    parser.add_argument('--prompt_post_pipline', type=str, default=prompt, help='Prompt used for data generation post pipline') 
    parser.add_argument('--data_read_path', type=str, default="datas/wizard_instruction/example.jsonl", help='Path for data reading')  
    parser.add_argument('--file_type', type=str, default="jsonl", help='data file type')  
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
    parser.add_argument('--need_post_pipeline_label_list', type=list, default=["BMI", "预期孕周", "药品"], help='List of labels requiring multi-stage pipeline processing') 
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

python examples/model_votes/model_votes_and_pipline.py

更多样例可参考文件夹：[examples/ ](https://github.com/jdh-algo/JoyDataForge/tree/main/examples "examples")中的示例内容。

## 数据集

我们利用[JoyDataForge](https://github.com/jdh-algo/JoyDataForge "数据合成 锻造厂")工具中的[Magpie](https://arxiv.org/abs/2406.08464 "喜鹊")方法和[Wizard](https://arxiv.org/abs/2304.12244 "WizardLM")方法合成了2个数据集，并进行了开源， 供大家参考和使用。

数据集1：[joy_common_sft_datasets](https://huggingface.co/datasets/jdh-algo-huggingface/joy_common_sft_datasets "magpie")， 数据集是利用magpie 方法对[Qwen2-7b-Intruction](https://github.com/QwenLM/Qwen)模型进行蒸馏， 可以用于模型**初期**的SFT的指令微调阶段。

数据集2：[query_diversity_evolution](https://huggingface.co/datasets/jdh-algo-huggingface/query_diversity_evolution "WizardLM") ， 本数据集是利用wizard方法对给定数据集中的query 内容进行深度和宽度的演化2轮得到结果， 我们并对演化后的query进行了任务难度打分， 用于提高query的多样性及任务的复杂度，用于模型的**后阶段**的SFT指令微调阶段。

## 贡献

描述如何为项目做出贡献，包括报告错误、提出功能建议或提交代码。

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/fooBar`)
3. 提交更改 (`git commit -am 'Add some fooBar'`)
4. 推送到分支 (`git push origin feature/fooBar`)
5. 创建新的 Pull Request

## 引用

```
@misc{joydataforge2024,
  title={JoyDataForge: An Open Source Toolkit for Large-Scale Data Synthesis},
  author={weili36190429 and lb2158},
  year={2024},
  howpublished={\url{https://github.com/jdh-algo/JoyDataForge.git}},
  note={Version 1.0}
}
```

## 项目贡献者

 ``<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
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
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/allwellll"><img src="https://avatars.githubusercontent.com/allwellll?s=100" width="100px;" alt="User's Name"/><br /><sub><b>User's Name</b></sub></a><br /><a href="#example-allwellll" title="Examples">💡</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/wanyueli"><img src="https://avatars.githubusercontent.com/wanyueli?s=100" width="100px;" alt="User's Name"/><br /><sub><b>User's Name</b></sub></a><br /><a href="#example-wanyueli" title="Examples">💡</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/stalwart0465"><img src="https://avatars.githubusercontent.com/stalwart0465?s=100" width="100px;" alt="User's Name"/><br /><sub><b>User's Name</b></sub></a><br /><a href="#content-stalwart0465" title="Content">🖋</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/tabhitmy"><img src="https://avatars.githubusercontent.com/tabhitmy?s=100" width="100px;" alt="User's Name"/><br /><sub><b>User's Name</b></sub></a><br /><a href="#content-tabhitmy" title="Content">🖋</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Hansen06"><img src="https://avatars.githubusercontent.com/Hansen06?s=100" width="100px;" alt="User's Name"/><br /><sub><b>User's Name</b></sub></a><br /><a href="#content-Hansen06" title="Content">🖋</a></td>
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

## 联系方式

- 邮箱: liwei1559@jd.com
