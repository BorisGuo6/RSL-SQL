"""
Step 3: Binary Selection Strategy (BSS)
二元选择策略

本步骤实现论文中的 Step 3 - 二元选择策略（Binary Selection Strategy, BSS）
主要功能：
1. 执行两个 SQL：分别执行 SQL_1（完整模式）和 SQL_2（简化模式），得到执行结果 R_1、R_2
2. 最优选择：使用 LLM 比较两个 SQL 及其执行结果，分析哪个更符合问题语义
3. 风险对冲：通过选择策略平衡完整模式（保留结构但可能有冗余）和简化模式（减少噪声但可能遗漏信息）的优缺点
4. 生成 SQL_3：选择更优的 SQL 作为 SQL_3

输出：
- step_3_binary.txt: 每个问题对应的最优 SQL（SQL_3）

核心目标：降低负影响（x），提升正收益（y），通过选择策略平衡两种模式的优缺点
"""

from llm.Binary_GPT import GPT
import json
from tqdm import tqdm
from utils.util import execute_sql
from utils.simplified_schema import simplified, explanation_collection
import argparse


def prompt_construct(simple_ddl, ddl_data, foreign_key, explanation, ppl, sql1, sql2):
    """
    构建二元选择提示：包含两个 SQL 及其执行结果
    
    功能：构建用于二元选择的提示，包含简化模式、两个候选 SQL 及其执行结果
    对应论文：Section III-D - Binary Selection Strategy
    
    处理流程：
    1. 构建简化模式信息（包含列描述）
    2. 执行两个 SQL：
       - SQL_1（完整模式生成）→ 得到执行结果 R_1
       - SQL_2（简化模式生成）→ 得到执行结果 R_2
    3. 构建候选 SQL 信息：包含两个 SQL 及其执行结果
    
    参数：
        simple_ddl: 简化后的表结构
        ddl_data: 简化后的数据样本
        foreign_key: 外键信息
        explanation: 列描述信息
        ppl: 包含问题、数据库等信息的字典
        sql1: Step 1 生成的 SQL_1（完整模式）
        sql2: Step 2 生成的 SQL_2（简化模式）
        
    返回：
        table_info: 包含简化模式和问题的提示
        candidate_sql: 包含两个 SQL 及其执行结果的字符串
    """
    db = ppl['db']
    question = ppl['question'].strip()
    evidence = ppl['evidence'].strip()
    example = ppl['example']

    # 构建简化模式信息（包含列描述）
    table_info = '### Sqlite SQL tables, with their properties:\n'
    table_info += simple_ddl + '\n' + '### Here are some data information about database references.\n' + ddl_data + '\n### Foreign key information of Sqlite SQL tables, used for table joins:\n' + foreign_key + '\n### The meaning of every column:\n#\n' + explanation.strip() + "\n#\n"

    table_info = example.strip() + '\n\n' + "### Answer the question by sqlite SQL query only and with no explanation. You must minimize SQL execution time while ensuring correctness.\n" + table_info.strip() + '\n\n' + '### definition: ' + evidence + "\n### Question: " + question

    # 执行两个 SQL，获取执行结果
    # SQL_1（完整模式）的执行结果 R_1
    r1, c1, re1 = execute_sql(sql1, db)
    # SQL_2（简化模式）的执行结果 R_2
    r2, c2, re2 = execute_sql(sql2, db)

    # 构建候选 SQL 信息：包含两个 SQL 及其执行结果
    candidate_sql = f"### sql1: {sql1} \n### result1: {re1} \n### sql2: {sql2} \n### result2: {re2}"

    return table_info, candidate_sql


def sql_generation(table_info, candidate_sql):
    """
    二元选择：使用 LLM 选择最优 SQL
    
    功能：比较两个 SQL 及其执行结果，选择更符合问题语义的 SQL
    对应论文：Section III-D - Binary Selection Strategy
    输出：SQL_3 - 选择的最优 SQL
    
    处理流程：
    1. 使用 LLM（Binary_GPT）分析两个 SQL 及其执行结果
    2. LLM 基于以下信息进行选择：
       - 简化模式信息（S'、V'、D'）
       - 问题定义和用户问题
       - 两个 SQL 及其执行结果（R_1、R_2）
    3. LLM 判断哪个 SQL 更符合问题语义，返回最优 SQL
    
    核心优势：通过风险对冲，平衡完整模式（保留结构）和简化模式（减少噪声）的优缺点
    
    参数：
        table_info: 包含简化模式和问题的提示
        candidate_sql: 包含两个 SQL 及其执行结果的字符串
        
    返回：
        answer: 包含选择的最优 SQL 的字典
    """
    binary_gpt = GPT()
    # 调用 LLM 进行二元选择
    answer = binary_gpt(table_info, candidate_sql)
    try:
        answer = json.loads(answer)
    except Exception as e:
        print(e)
        # 处理转义字符问题
        answer = answer.replace("\\", "\\\\")
        answer = json.loads(answer)
    return answer


def main(ppl_file, output_file, sql_file1, sql_file2, x=0):
    """
    Step 3 主函数：处理所有问题，通过二元选择策略选择最优 SQL
    
    处理流程：
    1. 加载输入数据：
       - ppl_dev.json：包含问题、数据库等信息
       - preliminary_sql.txt：Step 1 生成的 SQL_1（完整模式）
       - step_2_information_augmentation.txt：Step 2 生成的 SQL_2（简化模式）
    2. 对每个问题执行以下步骤：
       a. 模式简化（simplified）→ 生成简化模式 S'、V'
       b. 列描述收集（explanation_collection）→ 生成列描述 D'
       c. 构建提示并执行 SQL（prompt_construct）：
          - 执行 SQL_1 → 得到执行结果 R_1
          - 执行 SQL_2 → 得到执行结果 R_2
       d. 二元选择（sql_generation）→ 使用 LLM 比较两个 SQL 及其结果，选择最优 SQL_3
       e. 保存结果到文件
    3. 输出文件：
       - output_file (step_3_binary.txt): 所有问题的最优 SQL（SQL_3）
    
    核心目标：通过风险对冲，降低负影响（x），提升正收益（y）
    
    参数：
        ppl_file: 输入文件路径（ppl_dev.json）
        output_file: 输出 SQL 文件路径（step_3_binary.txt）
        sql_file1: Step 1 的 SQL 文件路径（preliminary_sql.txt）
        sql_file2: Step 2 的 SQL 文件路径（step_2_information_augmentation.txt）
        x: 起始索引，用于断点续跑
    """
    # 1. 加载输入数据
    with open(ppl_file, 'r') as f:
        ppls = json.load(f)

    # 加载 Step 1 生成的 SQL_1（完整模式）
    with open(sql_file1, 'r') as f:
        sqls1s = f.readlines()

    # 加载 Step 2 生成的 SQL_2（简化模式）
    with open(sql_file2, 'r') as f:
        sqls2s = f.readlines()

    answers = []  # 存储所有选择的最优 SQL

    # 2. 遍历处理每个问题
    for i in tqdm(range(x, len(ppls))):
        ppl = ppls[i]
        sql1 = sqls1s[i].strip()  # SQL_1（完整模式）
        sql2 = sqls2s[i].strip()  # SQL_2（简化模式）

        # 2.1 模式简化：生成简化模式 S'、V'
        simple_ddl, ddl_data, foreign_key = simplified(ppl)

        # 2.2 列描述收集：生成列描述 D'
        explanation = explanation_collection(ppl)

        # 2.3 构建提示并执行 SQL
        # 功能：执行两个 SQL，获取执行结果 R_1、R_2
        # 输出：table_info（简化模式信息）、candidate_sql（两个 SQL 及其执行结果）
        table_info, candidate_sql = prompt_construct(simple_ddl, ddl_data, foreign_key, explanation, ppl, sql1, sql2)

        # 2.4 二元选择：使用 LLM 比较两个 SQL 及其执行结果，选择最优 SQL_3
        # 功能：基于简化模式、问题、两个 SQL 及其执行结果，让 LLM 选择更优的 SQL
        # 核心：通过风险对冲，平衡完整模式（保留结构）和简化模式（减少噪声）的优缺点
        sql = sql_generation(table_info, candidate_sql)
        answer = sql['sql'].replace('\n', ' ')
        answers.append(answer)

        # 2.5 实时保存结果（防止中断丢失数据）
        with open(output_file, 'w', encoding='utf-8') as file:
            for sql in answers:
                file.write(str(sql) + '\n')


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()

    # 添加命令行选项

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--ppl_file", type=str, default="src/information/ppl_dev.json")
    parser.add_argument("--sql_3_output", type=str, default="src/sql_log/step_3_binary.txt")
    parser.add_argument("--sql_1", type=str, default="src/sql_log/preliminary_sql.txt")
    parser.add_argument("--sql_2", type=str, default="src/sql_log/step_2_information_augmentation.txt")

    # 解析命令行参数
    args = parser.parse_args()

    main(args.ppl_file, args.sql_3_output, args.sql_1, args.sql_2, args.start_index)
