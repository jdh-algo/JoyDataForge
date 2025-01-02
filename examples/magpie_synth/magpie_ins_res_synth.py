"""
This module defines a script for generating synthetic data using the MagpieGenerator class.

The file includes the following import statements:
- Standard libraries: os, sys, time, argparse, asyncio, pathlib
- Third-party library: loguru for logging
- Custom module: MagpieGenerator from src.joydataforge.components.synth.magpie_synth.data_generate

The file also includes the following classes and functions:
- get_args(): Parses command-line arguments to configure the data generation process.
- main(): The main asynchronous function that initializes the MagpieGenerator and generates data.

To use this module, you can run it as a script with the appropriate command-line arguments to generate synthetic data and log the results.
"""
import os
import sys
from pathlib import Path
import time
import argparse
import asyncio
import torch
from loguru import logger
from src.joydataforge.components.synth.magpie_synth.data_generate import MagpieGenerator

sys.path.append(str(Path(__file__).parents[2]))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/mnt/afs2/zhy/cache/models/Qwen2-7B-Instruct")
    parser.add_argument("--engine", type=str, default="vllm", choices=["vllm"])
    parser.add_argument("--output_folder", type=str, default="results/magpie")
    parser.add_argument("--job_name", type=str, default="magpie_synth")
    parser.add_argument("--device", type=str, default="0,1,2,3")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--total_prompts", type=int, default=50000, help="Number of samples")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timestamp", type=int, default=int(time.time()))
    parser.add_argument("--control_tasks", type=str, default=None)
    parser.add_argument("--system_prompt", action="store_true")
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--skip_special_tokens", action="store_true")
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--logits_processor", action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--swap_space", type=int, default=4)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--generate_response_immediately", type=bool, default=True)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--magpie_model_templates_path", type=str, default="src/joydataforge/config/magpie_model_templates.json")

    return parser.parse_args()


async def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.tensor_parallel_size = 4

    # 验证 CUDA 是否可用
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")

    generator = MagpieGenerator(args)
    results = await generator.generate()
    logger.info(f"Generated {len(results)} instructions")
    logger.info(f"Results saved to {generator.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
