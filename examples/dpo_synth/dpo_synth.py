"""
This module defines a script for loading, processing, and generating synthetic data using various components.

The file includes the following import statements:
- Standard libraries: os, sys, argparse, asyncio, pathlib
- Third-party library: loguru for logging
- Custom modules:
  - DataLoadAndProcess from src.joydataforge.components.loader.data_load_and_process
  - DataCache from src.joydataforge.memory.cache.data_cache
  - JoyDataGenerate from src.joydataforge.components.synth.joy_synth.data_generate
  - model_args from examples.dpo_synth.model_args_parser
  - JoyLLM from src.joydataforge.models.llm

The file also includes the following classes and functions:
- get_args(): Parses command-line arguments to configure the data loading, processing, and generation process.
- main(): The main asynchronous function that sets up the environment, processes data, and generates synthetic data.

To use this module, you can run it as a script with the appropriate command-line arguments to process input data and generate synthetic data, saving the results to the specified location.
"""

import os
import sys
from pathlib import Path
import argparse
import asyncio
from loguru import logger
from src.joydataforge.components.loader.data_load_and_process import DataLoadAndProcess
from src.joydataforge.memory.cache.data_cache import DataCache
from src.joydataforge.components.synth.joy_synth.data_generate import JoyDataGenerate
from examples.dpo_synth.model_args_parser import model_args
from src.joydataforge.models.llm import JoyLLM

sys.path.append(str(Path(__file__).parents[2]))
logger.info(f"model_args: {model_args}")


def get_args():
    parser = argparse.ArgumentParser()
    # Business configuration for DPO scoring
    parser.add_argument('--data_write_path', type=str, default="results/dpo", help='Location to save generated data')
    parser.add_argument('--data_write_file_name', type=str, default="dpo_data_synth.json", help='File name for saving generated data')
    parser.add_argument('--data_read_path', type=str, default="datas/dpo/dpo_scoring_test_new.json",
                        help='Path to the reference data source, can be a path or a file')
    parser.add_argument('--file_type', type=str, default="json", help='File format of the source data')
    parser.add_argument('--method', type=str, default="dpo_reward_model_scoring",
                        choices=['default', 'vote', 'vote_and_pipeline', 'all_query_intentions', "wizard_evolution",
                                 "dpo_reward_model_scoring", "case_rewrite", "synthesis_directly"],
                        help='Data generation method, coarse and fine filtering to improve accuracy')
    parser.add_argument('--task_name', type=str, default="dpo_scoring", help='Task name')
    parser.add_argument('--from_where', type=str, default="internet", help='Data source')
    parser.add_argument('--threshold', type=int, default=10, help='Upper limit of data processing volume')
    parser.add_argument('--threshold_low', type=int, default=0, help='Starting position of data processing')
    parser.add_argument('--max_works', type=int, default=1, help='Number of parallel processing workers')
    parser.add_argument('--is_local_rm_mode', type=bool, help='Whether it is DPO scoring for model responses')
    parser.add_argument('--is_save_caches', type=bool, help='Whether to save caches')

    return parser.parse_args()


async def main():
    args = get_args()
    data_path_destination = args.data_write_path
    if not os.path.exists(data_path_destination):
        os.makedirs(data_path_destination)

    # Locally deployed reward model
    local_model = JoyLLM(args=model_args, is_local_rm_mode=True, engine="hf")
    # Data that needs to be removed or deduplicated
    caches = DataCache()
    logger.info(f"cache_size: {len(caches.cache)}")
    # Data pre-processing
    data_processor = DataLoadAndProcess(path=args.data_read_path, task=args.task_name, file_type=args.file_type)
    # Data generation
    output_file = data_path_destination + f'/{args.data_write_file_name}_{args.task_name}.json'
    data_generator = JoyDataGenerate(
        model=local_model,
        output_file=output_file,
        method=args.method,
        threshold=args.threshold,
        threshold_low=args.threshold_low,
        max_works=args.max_works,
        from_where=args.from_where,
        task_name=args.task_name,
        caches=caches,
        data_processor=data_processor
    )
    # Perform intention recognition on all queries
    await data_generator.generate()
    if args.is_save_caches:
        cache_save_path = "caches/cache_find_all_intentions.json"
        caches.save_cache(save_path=cache_save_path)
    logger.info(f"\nDataset: {args.data_read_path} processing completed!")


if __name__ == "__main__":
    asyncio.run(main())
