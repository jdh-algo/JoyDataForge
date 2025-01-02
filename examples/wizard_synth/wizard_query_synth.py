"""
This module defines the "Wizard Data Synthesis" process.

The file includes the following import statements:
- os
- sys
- asyncio
- argparse
- Path from pathlib
- logger from loguru
- gpt4o_mini and JoyLLM from src.joydataforge.models.llm
- DataLoadAndProcess from src.joydataforge.components.loader.data_load_and_process
- DataCache from src.joydataforge.memory.cache.data_cache
- JoyDataGenerate from src.joydataforge.components.synth.joy_synth.data_generate

The file also includes the following classes and functions:
- get_args(): parses command-line arguments
- main(): the main function for running the Wizard Data Synthesis process

To use this module, you can run the script with the desired command-line arguments.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
from loguru import logger
from src.joydataforge.models.llm import gpt4o_mini, JoyLLM
from src.joydataforge.components.loader.data_load_and_process import DataLoadAndProcess
from src.joydataforge.memory.cache.data_cache import DataCache
from src.joydataforge.components.synth.joy_synth.data_generate import JoyDataGenerate



def get_args():
    parser = argparse.ArgumentParser(description='Use the Wizard method for data synthesis')
    parser.add_argument('--model', type=JoyLLM, default=gpt4o_mini, help='Model used for data generation')
    # parser.add_argument('--output_file', type=str, help='File for storing the generated data')  
    parser.add_argument('--data_read_path', type=str, default="datas/wizard_instruction",
                        help='Path to the reference data source, can be a path or a file')
    parser.add_argument('--file_type', type=str, default="jsonl", help='File format of the source data')
    parser.add_argument('--read_file_name', type=str, default="wizard_synth_datas_filtered.json", help='File name of the source data')
    parser.add_argument('--data_write_path', type=str, default="results/wizard_instruction", help='Path for saving the generated data')
    parser.add_argument('--data_write_file_name', type=str, default="wizard_evolution_data_synth.json",
                        help='File name for saving the generated data')
    parser.add_argument('--task_name', type=str, default="wizard", help='Task name')
    parser.add_argument('--from_where', type=str, default="***", help='Data source')
    parser.add_argument('--method', type=str,
                        choices=['default', 'vote', 'vote_and_pipeline', 'all_query_intentions', "wizard_evolution",
                                 "dpo_reward_model_scoring", "case_rewrite", "synthesis_directly"], default="wizard_evolution",
                        help='Data production method')
    parser.add_argument('--threshold', type=int, default=5000, help='Upper limit for processing data quantity')
    parser.add_argument('--threshold_low', type=int, default=0, help='Starting position for processing data')
    parser.add_argument('--max_works', type=int, default=10, help='Number of parallel processing works')
    parser.add_argument('--wizard_evolution_round', type=int, default=2, help='Wizard evolution round threshold')
    parser.add_argument('--data_cache_path', type=str, default="",
                        help='Path to already generated historical data (data that should not be regenerated), can be empty for the first generation')
    parser.add_argument('--is_save_caches', type=str, default=False,
                        help='Whether to save the generated data as historical cache data!')
    parser.add_argument('--save_cache_path', type=str, default="caches/caches.json", help='Path for saving cache data')
    parser.add_argument('--use_history', type=bool, default=False,
                        help='is use history messages, if for multi-turn you should set it to true')
    base_args = parser.parse_args()
    return base_args


async def main():
    args = get_args()

    data_path_destination = args.data_write_path
    if not os.path.exists(data_path_destination):
        os.makedirs(data_path_destination)
        # Data to be excluded or deduplicated
    caches = DataCache(args.data_cache_path)
    logger.info(f"cache_size:{len(caches.cache)}")
    # Data preprocessing
    data_pre_processor = DataLoadAndProcess(path=args.data_read_path, task=args.task_name, file_type=args.file_type)
    # Data generation
    output_file = data_path_destination + f'/{args.data_write_file_name}_{args.task_name}.json'
    data_generator = JoyDataGenerate(
        model=gpt4o_mini,
        output_file=output_file,
        method=args.method,
        caches=caches,
        threshold=args.threshold,
        threshold_low=args.threshold_low,
        max_works=args.max_works,
        from_where=args.from_where,
        task_name=args.task_name,
        data_processor=data_pre_processor,
        use_history=args.use_history
    )
    await data_generator.generate()
    if args.is_save_caches:
        caches.save_cache(save_path=args.save_cache_path)
    logger.info(f"\nDataset: {args.data_read_path} processed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
