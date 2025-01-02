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
    parser.add_argument('--save_cache_path', type=str, default="caches/caches.json", help='Path to save the cache data')
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
    data_processor = DataLoadAndProcess(path=args.data_read_path, task=args.task_name, file_type=args.file_type)
    
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
        data_processor=data_processor
    )
    
    await data_generator.generate()
    
    # After processing, data can be cached or used to rebuild
    if args.is_save_caches:
        caches_1.save_cache(save_path=args.save_cache_path)
    logger.info(f"Dataset: {args.data_read_path}, labeling processing completed!")

if __name__ == '__main__':
    asyncio.run(main())
