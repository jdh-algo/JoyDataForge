"""
This module defines a script for data sampling and evaluation, particularly for Chinese text classification tasks.

The file includes the following import statements:
- Standard libraries: os, sys, json, asyncio, argparse, pathlib
- Third-party libraries: numpy for numerical operations, loguru for logging
- Custom module: DataSamplingAndEvaluation from src.joydataforge.components.score.data_sampling

The file also includes the following functions:
- get_args(): Parses command-line arguments for configuring data sampling and evaluation.
- main(): The main asynchronous function that loads data, performs sampling, and evaluates the diversity of the selected data.

To use this module, you can run it as a script with the appropriate command-line arguments to perform data sampling and evaluation, saving the results to the specified location.
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
import json
import asyncio
import argparse
import numpy as np
from loguru import logger
from src.joydataforge.components.score.data_sampling import DataSamplingAndEvaluation



def get_args():
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--embeddings_path', type=str, default="datas/agent_intent/embeddings_examples.json",
                        help='Path to data embeddings')
    parser.add_argument('--data_read_path', type=str, default="datas/agent_intent/examples.json",
                        help='Path to the reference data source, can be a path or a file')
    parser.add_argument('--data_write_path', type=str, default="results/data_select_and_evaluation",
                        help='Location to save generated data')
    parser.add_argument('--data_write_file', type=str, default="evaluation_res.json", help='File name for saving generated data')
    parser.add_argument('--task_name', type=str, default="data_select_and_evaluation",
                        help='Task name for data sampling and evaluation')
    parser.add_argument('--select_methods', type=list, default=["k_center", "random"], help="Data sampling methods")
    parser.add_argument('--eval_methods', type=list, default=["vendi", "entity_diversity"], help="Data diversity evaluation methods")
    parser.add_argument('--is_need_embedding', type=bool, default=False,
                        help="Whether data needs embedding; if needed, specify the embedding model and method. Refer to [src/joydataforge/models/llm.EmbeddingModel]")
    parser.add_argument('--data_num_th', type=int, default=10, help="Number of data samples to select")
    base_args = parser.parse_args()
    return base_args


async def main():
    """Main function example"""
    args = get_args()
    embeddings = json.load(open(args.embeddings_path, "r", encoding="utf-8"))[:args.data_num_th]
    datas = open(args.data_read_path, "r", encoding="utf-8").readlines()[:args.data_num_th]

    # Create an instance of the sampler
    sampler = DataSamplingAndEvaluation(
        data=datas,
        embedding=np.array(embeddings),
        select_nums=args.data_num_th,
        select_methods=args.select_methods,
        eval_methods=args.eval_methods,
        need_embedding=args.is_need_embedding
    )
    final_result = {}
    if not os.path.exists(args.data_write_path):
        os.makedirs(args.data_write_path)

    # Use a context manager for sampling
    async with sampler.sampling_session():
        results = await sampler.sampling(output_dir=args.data_write_path, num_samples=args.data_num_th)
        for method in args.select_methods:
            # Get selected data
            selected_data = await sampler.get_selected_data(method=method, results=results)
            # Get evaluation results
            eval_results = await sampler.get_evaluation_results(method=method, results=results)
            logger.info(f"Selected data: {len(selected_data)}")
            logger.info(f"Evaluation results: {eval_results}")
            final_result.setdefault(method, {}).setdefault("num", len(selected_data))
            final_result.setdefault(method, {}).setdefault("scores", eval_results)

    with open(os.path.join(args.data_write_path, args.data_write_file), "w", encoding="utf-8") as wf:
        json.dump(final_result, wf, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
