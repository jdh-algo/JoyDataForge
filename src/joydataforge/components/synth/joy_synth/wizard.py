"""
This module defines the functionality for creating wizard prompts and processing tasks.

The file includes the following import statements:
- datetime
- List from typing
- logger from loguru
- configuration prompts from src.joydataforge.config

The file also includes the following classes and functions:
- create_wizard_prompt
- flatten_list
- generate_save_version
- createBreadthPrompt
- createDeepenPrompt
- createComplexInstruction
- result_post_process
- convert_query_format

To use this module, you can import the necessary functions and call them with appropriate parameters.
"""

import datetime
from typing import List
from loguru import logger
from src.joydataforge.config import wizaed_base_breadth_instruction_prompt, wizaed_base_depth_instruction_prompt, \
    wizaed_base_diffculty_judge_prompt

D_TASK_MAP = {"B": "wizard_breadth_querys", "D": "wizard_depth_querys"}
D_TASK_OUTPUT_TYPE = {"B": "json", "D": "json", "C": "json"}


async def create_wizard_prompt(item, prompt_type, wizard_query="", task_label="", use_history=False):
    """
    This function is used to synthesize breadth/depth prompts
    """
    format_item = await convert_query_format(item, wizard_query=wizard_query, task_label=task_label, use_history=use_history)
    if format_item.get("code", -1) == 0 and format_item.get("task_label"):
        instruction = [item.get("content", "") for item in format_item.get("data", []) if item.get("role", "") == "user"][0]
        task_label = format_item.get("task_label", [])[0]

        if prompt_type == "B":
            prompt = await create_breadth_prompt(instruction=instruction, task_label=task_label)
        elif prompt_type == "D":
            prompt = await create_deepen_prompt(instruction=instruction, task_label=task_label)
        elif prompt_type == "C":
            prompt = await create_complex_instruction(instruction=instruction)
    else:
        prompt = ""

    return prompt


async def flatten_list(_list: List) -> List:
    """
    This function flattens nested lists to return non-nested data.
    """

    def flatten(_list):
        for _data in _list:
            if hasattr(_data, "__iter__") and not isinstance(_data, str):
                for sub in flatten(_data):
                    yield sub
            else:
                yield _data

    return [x for x in flatten(_list)]


async def generate_save_version():
    """
    This function generates the final training version based on the input data version and the current date
    """
    now = datetime.datetime.now()
    save_version = f"save_version_{now.year}{now.month:02d}{now.day:02d}-{now.hour:02d}{now.minute:02d}{now.second:02d}"
    return save_version


async def create_breadth_prompt(instruction, task_label):
    """
    This function is used to expand the diversity of queries in a breadth-oriented manner.
    """
    prompt = wizaed_base_breadth_instruction_prompt
    prompt += "#给定的提示#: \r\n {} \r\n".format(instruction)
    prompt += "#分类#: \r\n {} \r\n".format(task_label)
    prompt += """### 注意请严格按照[“xxx”, "xxx", "xxx"]的格式输出，前后不要带有任何多余的字符串"""
    return prompt


async def create_deepen_prompt(instruction, task_label):
    """
    This function is used to expand the difficulty of queries in a depth-oriented manner.
    """
    prompt = wizaed_base_depth_instruction_prompt.format(
        """如果#给定的提示#包含对某些问题的探询，改变探询的方向或者范围。你需要根据提供的#给定的提示#, 按照下列的方式思考。
        step1:要求选择的新类别必须和原有的#分类#不同，但是不会出现较大领域的变更（如从医学问题变为化学问题），可行且合理。例如可以从类别"导购-药品名产品名"变为"合理用药-适应症禁忌症科普"，但是不能变为"症状咨询"，因为症状咨询需要中只包含症状描述，而不涉及药品。
        step2:在保证与原问题相关的前提下，思考如何改变问题的探询方向或范围到新类别。
        step3:请对你思考的领域进行深入思考，使得创建的问题具备一定的难度（需要更多的的参考知识、更多的逻辑推理步骤）才能解决
        step4:对#创建的提示#进行逻辑判断，是否符合常人的提问, 是否不包含不明的指代信息（如“以上报告”、“这个药”等不明指代）,在保证新的问题在单独拿出来之后可以脱离上下文被正常人理解之后输出结果
        final:输出3个创建的提示和新类别，输出格式要求为可被python解析的格式（列表里面包含3个json, json里面有query和changeType两个字段），生成的结果不应该包含"json"和"```"。\r\n
        参考如下:

        示例1:
        #给定的提示#:最近老是觉得手脚冰凉，怎么回事？
        #分类#:症状咨询
        思考:原始提示询问的是个人症状的原因,类别为:症状咨询，因此考虑从该症状相关的潜在健康问题、潜在病因、解决方案建议三个角度出发进行问题的拓展与深化
        输出:[{"input":"手脚冰凉可能与哪些潜在健康问题有关？", "changeType":"疾病咨询"},{"input":"手脚冰凉是否与血液循环问题有关？", "changeType":"生理机制咨询"},{"input":"改善手脚冰凉有哪些生活方式建议？", "changeType":"健康管理咨询"}]\r\n

        示例3:
        #给定的提示#:儿童腹泻是否需要进行饮食调整或补液治疗？
        #分类#: 治疗方案咨询
        思考: 原始提示询问的是儿童的腹泻治疗方案，这通常可以从营养管理、家庭护理、健康监测等角度展开
        输出:[{"input": "儿童腹泻期间如何调整饮食以促进康复？", "changeType": "营养管理咨询"},{"input": "有哪些常用的家庭护理方法可以帮助缓解儿童腹泻？", "changeType": "家庭护理建议"},{"input": "如何判断儿童腹泻时是否需要补液？", "changeType": "健康监测咨询"}]\r\n
        """
    )
    prompt += "take a deep breath and work on this problem step-by-step:"
    prompt += "#给定的提示#: \r\n{} \r\n".format(instruction)
    prompt += "#分类#: \r\n{} \r\n".format(task_label)
    prompt += """请严格按照[{"input": xxx, "changeType":xxx},{"input": xxx, "changeType":xxx},{"input": xxx, "changeType":xxx}]的格式输出，前后不要带有任何多余的字符串"""
    return prompt


async def create_complex_instruction(instruction):
    prompt = wizaed_base_diffculty_judge_prompt.format(instruction=instruction)
    return prompt


async def result_post_process(data, task_type):
    if task_type == "B":
        data = data.split("#-#")
        data = [d.strip("#").strip("-").split("\n") for d in data]
        data = await flatten_list(data)
    return data


async def convert_query_format(item, use_rewrite_query=True, use_history=True, wizard_query="", task_label=""):
    history = item.get("history", [])
    prefix = item.get("prefix", "")
    if not task_label:
        task_label = item.get("task_label", "")
    if not task_label:
        logger.error(
            "No task label type specified in input parameters and data! The wizard cannot evolve to generate corresponding data!")

    # 1. 如果能够拿到wizard合成的query, 则不带历史上文
    # 2. 如果能拿到rewrite_query，则带上历史上文
    # 3. 如果以上query都拿不到，则默认取原文query
    if wizard_query:
        input = wizard_query
        use_history = False
    elif use_rewrite_query:
        input = item.get("rewrite_query", "")

    if not input:
        input = item.get("input", "")
        logger.info(f'Using original query!')

    if not input:
        logger.error(f"Input data is empty")
        return dict(code=1, data=dict(), task_label="", wizard_llm_querys=[])

    if use_history:
        new_input = "\n[历史对话]：\n"
        for idx, his in enumerate(history):
            q = f"[Round-{idx}]问：" + his[0] + '\n'
            a = f"[Round-{idx}]答：" + his[1] + '\n'
            new_input += q
            new_input += a
        new_input += f"[用户最新问题]：\n{input}"
    else:
        new_input = input

    sample = [{"role": "user", "content": new_input},
              {"role": "assistant", "content": item["target"]}]

    return dict(code=0, data=sample, task_label=task_label, wizard_llm_querys=[])
