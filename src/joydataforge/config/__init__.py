import os
import sys
from pathlib import Path

from dynaconf import Dynaconf

_BASE_DIR = Path(__file__).parent.parent

prompt_settings_files = [
    Path(__file__).parent / 'prompt.yaml',
    ]  # 指定绝对路径加载默认配置

prompt_configs = Dynaconf(
    envvar_prefix="DATA_FORGE_PROMPT",  # 环境变量前缀
    settings_files=prompt_settings_files,
    environments=False,  # 启用多层次日志，支持 dev, pro
    load_dotenv=True,  # 加载 .env
    env_switcher="DATA_FORGE_PROMPT_ENV",  # 用于切换模式的环境变量名称 
    lowercase_read=True,  # 禁用小写访问， settings.name 是不允许的
    includes=[os.path.join(sys.prefix, 'export', 'JoyDataForge', 'settings.yml')],  # 自定义配置覆盖默认配置
    base_dir=_BASE_DIR,  # 编码传入配置
)
#agent
agent_cdss_one_qa_labeling_prompt=prompt_configs.agent.cdss.one_qa_labeling_prompt
agent_cdss_all_qa_labeling_prompt=prompt_configs.agent.cdss.all_qa_labeling_prompt
agent_cdss_one_qa_for_pregnancy_labeling_prompt=prompt_configs.agent.cdss.one_qa_for_pregnancy_labeling_prompt
agent_cdss_one_qa_for_medicine_labeling_prompt=prompt_configs.agent.cdss.one_qa_for_medicine_labeling_prompt
agent_cdss_one_qa_for_medicine_synthesis_prompt=prompt_configs.agent.cdss.one_qa_for_medicine_synthesis_prompt
agent_cdss_one_qa_for_pregnancy_synthesis_prompt=prompt_configs.agent.cdss.one_qa_for_pregnancy_synthesis_prompt

#wizard
wizaed_base_breadth_instruction_prompt=prompt_configs.wizardllm.Breadth_base_instruction
wizaed_base_depth_instruction_prompt=prompt_configs.wizardllm.base_Depth_instruction
wizaed_base_diffculty_judge_prompt=prompt_configs.wizardllm.diffculty_judge_prompt



model_settings_files = [
    Path(__file__).parent / 'model.yaml',
    ]  # 指定绝对路径加载默认配置

model_configs = Dynaconf(
    envvar_prefix="DATA_FORGE_MODEL",  # 环境变量前缀
    settings_files=model_settings_files,
    environments=False,  # 启用多层次日志，支持 dev, pro
    load_dotenv=True,  # 加载 .env
    env_switcher="DATA_FORGE_MODEL_ENV",  # 用于切换模式的环境变量名称 
    lowercase_read=True,  # 禁用小写访问， settings.name 是不允许的
    includes=[os.path.join(sys.prefix, 'export', 'JoyDataForge', 'settings.yml')],  # 自定义配置覆盖默认配置
    base_dir=_BASE_DIR,  # 编码传入配置
)

base_model_name=model_configs.app.base.NAME
base_model_url=model_configs.app.base.URL
base_model_url_outside=model_configs.app.base.URL_OUTSIDE
base_model_app_key=model_configs.app.base.API_KEY


embedding_model_name=model_configs.app.embedding_model.NAME
embedding_model_url=model_configs.app.embedding_model.URL




