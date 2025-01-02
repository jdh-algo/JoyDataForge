"""
This module defines a script for generating and processing data, specifically for Chinese text classification tasks.

The file includes the following import statements:
- Standard libraries: os, sys, asyncio, argparse, pathlib
- Third-party library: loguru for logging
- Custom modules:
  - JoyLLM and gpt4o_mini from src.joydataforge.models.llm
  - agent_cdss_all_qa_labeling_prompt and agent_cdss_one_qa_for_medicine_labeling_prompt from src.joydataforge.config
  - DataLoadAndProcess from src.joydataforge.components.loader.data_load_and_process
  - DataCache from src.joydataforge.memory.cache.data_cache
  - JoyDataGenerate from src.joydataforge.components.synth.joy_synth.data_generate

The file also includes the following functions:
- get_args(): Parses command-line arguments for configuring data generation and processing.
- main(): The main asynchronous function that handles data loading, generation, sampling, and post-processing.

To use this module, you can run it as a script with the appropriate command-line arguments to generate and process data, saving the results to the specified location.
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
import asyncio
import argparse
from loguru import logger
from src.joydataforge.models.llm import gpt4o_mini, gpt4o, JoyLLM
from src.joydataforge.config import agent_cdss_all_qa_labeling_prompt as prompt
from src.joydataforge.config import agent_cdss_one_qa_for_medicine_labeling_prompt as prompt_binary_classify
from src.joydataforge.components.loader.data_load_and_process import DataLoadAndProcess
from src.joydataforge.memory.cache.data_cache import DataCache
from src.joydataforge.components.synth.joy_synth.data_generate import JoyDataGenerate



def get_args():
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--model', type=JoyLLM, default=gpt4o_mini, help='Model used for data generation')
    parser.add_argument('--prompt', type=str, default=prompt, help='Prompt used for data generation')
    parser.add_argument('--output_file', type=str, help='File to store the generated data')
    parser.add_argument('--data_read_path', type=str, default="datas/cdss_datas/examples.jsonl",
                        help='Path to the reference data source, can be a path or a file')
    parser.add_argument('--file_type', type=str, default="jsonl", help='File format of the source data')
    parser.add_argument('--read_file_name', type=str, default="examples.jsonl", help='File format of the source data')
    parser.add_argument('--data_write_path', type=str, default="results/cdss_datas", help='Location to save generated data')
    parser.add_argument('--data_write_file_name', type=str, default="cdss_agent_data_synth.json",
                        help='File name for saving generated data')
    parser.add_argument('--task_name', type=str, default="fz", help='Task name')
    parser.add_argument('--from_where', type=str, default="***", help='Data source')
    parser.add_argument('--method', type=str, default="all_query_intentions",
                        choices=['default', 'vote', 'vote_and_pipeline', 'all_query_intentions', "wizard_evolution",
                                 "dpo_reward_model_scoring", "case_rewrite", "synthesis_directly"], help='Data generation method')
    parser.add_argument('--threshold', type=int, default=10, help='Upper limit of data processing amount')
    parser.add_argument('--threshold_low', type=int, default=0, help='Starting position of data processing')
    parser.add_argument('--max_works', type=int, default=10, help='Number of parallel processing workers')
    parser.add_argument('--data_cache_path', type=str, default="",
                        help='Path to previously generated data cache (to avoid regenerating data), can be empty for the first generation')
    parser.add_argument('--is_save_caches', type=str, default=False, help='Whether to save generated data as historical cache data')
    parser.add_argument('--save_cache_path', type=str, default="caches/caches.json",
                        help='Path to save cache data, effective only when is_save_caches=True')

    # Data sampling
    parser.add_argument('--one_label_total_up_threshold', type=int, default=100,
                        help='Upper limit of data sampling for a single label in the entire dataset')
    parser.add_argument('--one_data_one_label_sampling_up_threshold', type=int, default=4,
                        help='Upper limit of sampling for a single label in a single data entry')
    parser.add_argument('--history_round', type=int, default=2, help='Length of historical dialogue that can be referenced')
    parser.add_argument('--version', type=str, default="v_0", help='Data version')
    parser.add_argument('--test_data_num_threshold', type=int, default=10, help='num threshold of sampling test datasets')

    # Model post-processing
    parser.add_argument('--is_binary_classify', type=bool, default=True,
                        help='Whether to perform binary classification post-processing to improve accuracy')
    parser.add_argument('--need_sampling_labels', type=list, default=["BMI", "孕期孕周", "药品"],
                        help='List of labels that require model post-processing for refined tagging')
    parser.add_argument('--prompt_binary_classify', type=str, default=prompt_binary_classify, help='Prompt used for data generation')
    base_args = parser.parse_args()
    return base_args


async def main():
    args = get_args()

    if not os.path.exists(args.data_write_path):
        os.makedirs(args.data_write_path)

    # Data that needs to be excluded or deduplicated
    caches = DataCache()
    logger.info(f"Cache size: {len(caches.cache)}")

    # Data preprocessing
    data_processor_post = DataLoadAndProcess(path=args.data_read_path, task=args.task_name, file_type=args.file_type)

    # Data labeling for user query's intention for the given multi-conversation data.
    output_file = os.path.join(args.data_write_path, args.data_write_file_name)
    data_generator = JoyDataGenerate(
        model=args.model,
        prompt=args.prompt,
        output_file=output_file,
        method=args.method,
        caches=caches,
        threshold=args.threshold,
        threshold_low=args.threshold_low,
        max_works=args.max_works,
        from_where=args.from_where,
        task_name=args.task_name,
        data_processor=data_processor_post
    )

    # Perform intent recognition on all queries
    await data_generator.generate()

    if args.is_save_caches:
        caches.save_cache(save_path=args.save_cache_path)
    logger.info(f"\nDataset: {args.data_read_path} labeling completed!")

    # Sampling from the data with all intentions, specifying labels and thresholds as parameters
    logger.info(f"\nDataset: {output_file} starting sampling!")
    data_generator.sampling_from_data_with_all_intentions(
        data_path=output_file,
        save_path=args.data_write_path,
        test_file_name=f"test_dataset_{args.version}_{args.history_round}_middle.xlsx",
        train_file_name=f"train_dataset_{args.version}_{args.history_round}_middle.json",
        round=args.history_round,
        test_data_num_threshold=args.test_data_num_threshold,
        one_label_total_up_threshold=args.one_label_total_up_threshold,
        one_data_one_label_sampling_up_threshold=args.one_data_one_label_sampling_up_threshold,
        labels=args.need_sampling_labels
    )

    # If noise is found in the sampled data, specified labels can be filtered (e.g., binary classification for "药品" label to improve accuracy)
    data_file = args.data_write_path + f"/train_dataset_{args.version}_{args.history_round}_middle.json"
    logger.info(f"\nDataset: {data_file}, starting binary classification for fine annotation!")
    model_post_classify = gpt4o
    is_binary_classify = True
    output_file_post_binary = output_file + "_post_binary.json"
    if is_binary_classify:
        caches = DataCache()
        data_processor_post = DataLoadAndProcess(path=data_file, task=args.task_name, file_type="json")
        data_generator_post_binary = JoyDataGenerate(
            model=model_post_classify,
            prompt=args.prompt_binary_classify,
            output_file=output_file_post_binary,
            method="default",
            caches=caches,
            threshold=args.threshold,
            threshold_low=args.threshold_low,
            max_works=args.max_works,
            from_where=args.from_where,
            task_name=args.task_name,
            data_processor=data_processor_post
        )
        await data_generator_post_binary.generate()


if __name__ == "__main__":
    asyncio.run(main())
