"""
Step 2: SQL Generation based Simplified Schema and Information Augmentation (CIA)
基于简化模式的 SQL 生成与上下文信息增强

本步骤实现论文中的 Step 2 - 上下文信息增强（Contextual Information Augmentation, CIA）
主要功能：
1. 模式简化：基于 Step 1 的模式链接结果，生成简化后的数据库模式（S'、V'、D'）
2. SQL 组件生成：
   - 表/列元素生成（H_E）：识别需要的表和列
   - SQL 关键词生成（H_K）：生成如 DISTINCT、GROUP BY 等关键词
   - 条件生成（H_C）：从问题中分解出 WHERE 条件
3. 上下文增强：合并列描述 D' 与生成的组件信息（H_Aug = {D', H_E, H_C, H_K}）
4. SQL 生成：基于简化模式和增强信息生成 SQL_2

输出：
- step_2_information_augmentation.txt: 每个问题对应的增强 SQL（SQL_2）
- augmentation.json: 生成的上下文增强信息（包含表、列、关键词、条件等）

核心目标：提升正收益（y），通过增强信息帮助 LLM 更好理解简化模式
"""

from llm.LLM import GPT as model
import json
from tqdm import tqdm
from utils.simplified_schema import simplified, explanation_collection
from configs.Instruction import TABLE_AUG_INSTRUCTION, KEY_WORD_AUG_INSTRUCTION, CONDITION_AUG_INSTRUCTION, \
    SQL_GENERATION_INSTRUCTION
import argparse


def table_info_construct(simple_ddl, ddl_data, foreign_key, explanation):
    """
    构建简化后的数据库表信息提示
    
    功能：基于简化模式构建表信息，包含列描述（D'）
    对应论文：Section III-C - Contextual Information Augmentation
    
    与 Step 1 的区别：
    - 使用简化后的模式（simple_ddl, ddl_data）而非完整模式
    - 添加列描述（explanation），帮助 LLM 理解每列的含义
    
    参数：
        simple_ddl: 简化后的表结构（S'）
        ddl_data: 简化后的数据样本（V'）
        foreign_key: 外键信息
        explanation: 列描述信息（D'）
        
    返回：
        table_info: 格式化后的简化数据库表信息字符串
    """
    table_info = ('### Sqlite SQL tables, with their properties:\n' + simple_ddl +
                  '\n### Here are some data information about database references.\n' + ddl_data +
                  '\n### Foreign key information of Sqlite SQL tables, used for table joins:\n' + foreign_key +
                  '\n### The meaning of every column:\n#\n' + explanation.strip() +
                  '\n#\n')

    return table_info


def table_augmentation(table_info, ppl):
    """
    表/列元素生成（H_E）
    
    功能：识别简化模式中需要的表和列元素
    对应论文：Section III-C1 - SQL Components Generation (Elements)
    输出：H_E - 需要的表和列列表
    
    处理流程：
    1. 基于简化模式和问题，让 LLM 识别需要的表和列
    2. 与 Step 1 的前向链接类似，但基于简化模式而非完整模式
    
    参数：
        table_info: 简化后的数据库表信息
        ppl: 包含问题、证据等信息的字典
        
    返回：
        table_gpt_res: 包含 'tables' 和 'columns' 的字典（H_E）
    """
    question = ppl['question'].strip()
    evidence = ppl['evidence'].strip()

    gpt = model()
    table_gpt_res_prompt = table_info.strip() + '\n\n' + '### definition: ' + evidence + "\n### Question: " + question
    table_gpt_res = gpt(TABLE_AUG_INSTRUCTION, table_gpt_res_prompt)
    table_gpt_res = json.loads(table_gpt_res)
    return table_gpt_res


def key_word_augmentation(table_info, ppl):
    """
    SQL 关键词生成（H_K）
    
    功能：从问题中识别需要的 SQL 关键词（如 DISTINCT、GROUP BY、ORDER BY 等）
    对应论文：Section III-C1 - SQL Components Generation (Keywords)
    输出：H_K - SQL 关键词列表
    
    处理流程：
    1. 分析问题中的关键指示词（如"不同"→DISTINCT，"分组"→GROUP BY）
    2. 让 LLM 生成可能需要的 SQL 关键词
    
    参数：
        table_info: 简化后的数据库表信息
        ppl: 包含问题、证据等信息的字典
        
    返回：
        word_gpt_res: 包含 'sql_keywords' 的字典（H_K）
    """
    gpt = model()

    question = ppl['question'].strip()
    evidence = ppl['evidence'].strip()
    word_gpt_res_prompt = table_info.strip() + '\n\n' + '### definition: ' + evidence + "\n### Question: " + question
    word_gpt_res = gpt(KEY_WORD_AUG_INSTRUCTION, word_gpt_res_prompt)

    word_gpt_res = json.loads(word_gpt_res)
    return word_gpt_res


def condition_augmentation(ppl):
    """
    条件生成（H_C）
    
    功能：从问题中分解出 WHERE 子句可能需要的条件
    对应论文：Section III-C1 - SQL Components Generation (Conditions)
    输出：H_C - 条件列表
    
    处理流程：
    1. 分析用户问题，分解出可能的查询条件
    2. 让 LLM 生成 WHERE 子句中可能需要的条件表达式
    
    参数：
        ppl: 包含问题等信息的字典
        
    返回：
        relation_gpt_res: 包含 'conditions' 的字典（H_C）
    """
    gpt = model()

    question = ppl['question'].strip()
    relation_gpt_res = gpt(CONDITION_AUG_INSTRUCTION, question)
    relation_gpt_res = json.loads(relation_gpt_res)
    return relation_gpt_res


def sql_generation(ppl, table_aug, word_aug, cond_aug, table_info):
    """
    基于简化模式和上下文增强信息生成 SQL（SQL_2）
    
    功能：使用简化模式和上下文增强信息（H_Aug）生成 SQL
    对应论文：Section III-C2 - SQL Generation with Simplified Schema
    输出：SQL_2 - 基于简化模式和增强信息生成的 SQL
    
    处理流程：
    1. 合并上下文增强信息（H_Aug = {D', H_E, H_C, H_K}）：
       - D': 列描述（已在 table_info 中）
       - H_E: 表/列元素（table_aug）
       - H_K: SQL 关键词（word_aug）
       - H_C: 条件（cond_aug）
    2. 构建包含以下内容的提示：
       - 少样本示例（example）
       - 简化数据库模式（table_info，包含列描述 D'）
       - 上下文增强信息（H_E, H_K, H_C）
       - 问题定义和用户问题
    3. 调用 LLM 生成 SQL_2
    
    核心优势：通过上下文增强信息，帮助 LLM 更好理解简化模式，提升正收益（y）
    
    参数：
        ppl: 包含问题、示例等信息的字典
        table_aug: 表/列元素（H_E）
        word_aug: SQL 关键词（H_K）
        cond_aug: 条件（H_C）
        table_info: 简化后的数据库表信息（包含列描述 D'）
        
    返回：
        sql: 生成的 SQL 查询（SQL_2）
    """
    gpt = model()

    question = ppl['question'].strip()
    evidence = ppl['evidence'].strip()
    example = ppl['example']
    # 将上下文增强信息添加到表信息中
    # H_Aug = {D' (已在 table_info 中), H_E, H_C, H_K}
    table_info += f'\n### sql_keywords: {word_aug["sql_keywords"]}\n'  # H_K
    table_info += f'### tables: {table_aug["tables"]}\n'  # H_E
    table_info += f'### columns: {table_aug["columns"]}\n'  # H_E
    table_info += f'### conditions: {cond_aug["conditions"]}'  # H_C

    # 构建完整提示：示例 + 指令 + 简化模式 + 增强信息 + 问题
    table_info = example.strip() + '\n\n' + "### Answer the question by sqlite SQL query only and with no explanation. You must minimize SQL execution time while ensuring correctness.\n" + table_info.strip() + '\n\n' + '### definition: ' + evidence + "\n### Question: " + question

    # 调用 LLM 生成 SQL_2
    answer = gpt(SQL_GENERATION_INSTRUCTION, table_info)
    try:
        answer = json.loads(answer)
    except Exception as e:
        print(e)
        # 处理转义字符问题
        answer = answer.replace("\\", "\\\\")
        answer = json.loads(answer)
    # 清理 SQL：去除换行符
    sql = answer['sql'].replace('\n', ' ')
    return sql


def main(ppl_file, output_file, info_file, x):
    """
    Step 2 主函数：处理所有问题，生成基于简化模式的增强 SQL
    
    处理流程：
    1. 加载输入数据（ppl_dev.json，已包含 Step 1 的模式链接结果）
    2. 对每个问题执行以下步骤：
       a. 模式简化（simplified）→ 生成简化模式 S'、V'
       b. 列描述收集（explanation_collection）→ 生成列描述 D'
       c. 构建简化表信息（table_info_construct）→ 包含 D'
       d. SQL 组件生成：
          - 表/列元素生成（table_augmentation）→ H_E
          - SQL 关键词生成（key_word_augmentation）→ H_K
          - 条件生成（condition_augmentation）→ H_C
       e. SQL 生成（sql_generation）→ 使用 H_Aug = {D', H_E, H_C, H_K} 生成 SQL_2
       f. 保存结果到文件
    3. 输出文件：
       - output_file (step_2_information_augmentation.txt): 所有问题的增强 SQL（SQL_2）
       - info_file (augmentation.json): 上下文增强信息，供后续步骤使用
    
    核心目标：通过上下文信息增强，提升正收益（y），帮助 LLM 更好理解简化模式
    
    参数：
        ppl_file: 输入文件路径（ppl_dev.json，已包含模式链接结果）
        output_file: 输出 SQL 文件路径（step_2_information_augmentation.txt）
        info_file: 输出增强信息文件路径（augmentation.json）
        x: 起始索引，用于断点续跑
    """
    # 1. 加载输入数据：从 ppl_dev.json 读取所有问题（已包含 Step 1 的模式链接结果）
    with open(ppl_file, 'r') as f:
        ppls = json.load(f)

    answers = []  # 存储所有生成的增强 SQL
    informations = []  # 存储所有上下文增强信息

    # 2. 遍历处理每个问题
    for i in tqdm(range(x, len(ppls))):
        information = {}
        ppl = ppls[i]

        # 2.1 模式简化：基于 Step 1 的模式链接结果，生成简化模式
        # 功能：从完整模式中提取需要的表和列，生成简化模式 S'、V'
        # 调用：simplified(ppl) - 使用 ppl 中的模式链接结果
        simple_ddl, ddl_data, foreign_key = simplified(ppl)

        # 2.2 列描述收集：生成每列的含义描述
        # 功能：为简化模式中的每列生成文本描述，帮助 LLM 理解列的含义
        # 输出：D' - 列描述信息
        explanation = explanation_collection(ppl)

        # 2.3 构建简化表信息：包含列描述 D'
        # 功能：将简化模式、列描述等信息整合为提示
        table_info = table_info_construct(simple_ddl, ddl_data, foreign_key, explanation)

        # 2.4 SQL 组件生成 - 表/列元素（H_E）
        # 功能：识别简化模式中需要的表和列
        table_aug = table_augmentation(table_info, ppl)
        information['tables'] = table_aug['tables']
        information['columns'] = table_aug['columns']

        # 2.5 SQL 组件生成 - SQL 关键词（H_K）
        # 功能：从问题中识别需要的 SQL 关键词（如 DISTINCT、GROUP BY 等）
        word_aug = key_word_augmentation(table_info, ppl)
        information['sql_keywords'] = word_aug['sql_keywords']

        # 2.6 SQL 组件生成 - 条件（H_C）
        # 功能：从问题中分解出 WHERE 子句可能需要的条件
        cond_aug = condition_augmentation(ppl)
        information['conditions'] = cond_aug['conditions']
        informations.append(information)

        # 2.7 SQL 生成：基于简化模式和上下文增强信息生成 SQL_2
        # 功能：使用 H_Aug = {D', H_E, H_C, H_K} 生成 SQL
        # 核心：通过上下文增强信息，帮助 LLM 更好理解简化模式，提升正收益
        sql = sql_generation(ppl, table_aug, word_aug, cond_aug, table_info)

        answers.append(sql)

        # 2.8 实时保存结果（防止中断丢失数据）
        # 保存增强 SQL 到文件
        with open(output_file, 'w', encoding='utf-8') as file:
            for sql in answers:
                file.write(str(sql) + '\n')
        # 保存上下文增强信息到文件（供后续步骤使用）
        with open(info_file, 'w', encoding='utf-8') as file:
            json.dump(informations, file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()

    # 添加命令行选项

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--ppl_file", type=str, default="src/information/ppl_dev.json")
    parser.add_argument("--sql_2_output", type=str, default="src/sql_log/step_2_information_augmentation.txt")
    parser.add_argument("--information_output", type=str, default="src/information/augmentation.json")
    # 解析命令行参数
    args = parser.parse_args()

    main(args.ppl_file, args.sql_2_output, args.information_output, args.start_index)
