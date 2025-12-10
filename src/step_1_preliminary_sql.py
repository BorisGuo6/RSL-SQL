"""
Step 1: Preliminary SQL Generation and Bidirectional Schema Linking (BSL)
初步 SQL 生成与双向模式链接

本步骤实现论文中的 Step 1 - 双向模式链接（Bidirectional Schema Linking, BSL）
主要功能：
1. 前向模式链接（Forward Schema Linking）：使用完整数据库模式，让 LLM 识别与问题相关的表和列（L_fwd）
2. 初步 SQL 生成：基于完整模式生成 SQL_1，保留完整数据库结构，降低遗漏风险
3. 后向模式链接（Backward Schema Linking）：在 bid_schema_linking.py 中完成，解析 SQL_1 提取使用的表和列（L_bwd）
4. 模式简化：合并 L_fwd ∪ L_bwd 生成简化模式（在后续步骤中使用）

输出：
- preliminary_sql.txt: 每个问题对应的初步 SQL（SQL_1）
- LLM.json: 前向模式链接的结果（包含表、列信息），供后续后向链接使用
"""

from llm.LLM import GPT as model
import json
from tqdm import tqdm
from configs.Instruction import TABLE_AUG_INSTRUCTION, SQL_GENERATION_INSTRUCTION
import argparse


def table_info_construct(ppl):
    """
    构建完整的数据库表信息提示
    
    功能：从 ppl 数据中提取数据库结构信息，构建包含以下内容的提示：
    - 表结构（simplified_ddl）：SQLite 表的属性定义
    - 数据样本（ddl_data）：数据库中的数据信息，帮助理解列的含义
    - 外键信息（foreign_key）：用于表连接的外键关系
    
    参数：
        ppl: 包含问题、数据库结构等信息的字典
        
    返回：
        table_info: 格式化后的数据库表信息字符串，用于后续的 LLM 调用
    """
    (question, simple_ddl, ddl_data,
     foreign_key, evidence, example) = (ppl['question'].strip(), ppl['simplified_ddl'].strip(),
                                        ppl['ddl_data'].strip(), ppl['foreign_key'].strip(),
                                        ppl['evidence'].strip(), ppl['example'])

    table_info = ('### Sqlite SQL tables, with their properties:\n' + simple_ddl +
                  '\n### Here are some data information about database references.\n' + ddl_data +
                  '\n### Foreign key information of Sqlite SQL tables, used for table joins:\n' + foreign_key)
    return table_info


def table_column_selection(table_info, ppl):
    """
    前向模式链接（Forward Schema Linking, FSL）
    
    功能：使用完整数据库模式，让 LLM 识别与问题相关的表和列
    对应论文：Section III-B1 - Forward Schema Linking
    输出：L_fwd - 前向链接识别的表和列集合
    
    处理流程：
    1. 构建包含完整数据库模式、问题定义和用户问题的提示
    2. 调用 LLM（使用 TABLE_AUG_INSTRUCTION）识别相关表和列
    3. 解析 LLM 返回的 JSON 结果，提取表名和列名
    
    参数：
        table_info: 完整的数据库表信息
        ppl: 包含问题、证据等信息的字典
        
    返回：
        table_column: 包含 'tables' 和 'columns' 的字典，即前向链接结果 L_fwd
    """
    gpt = model()
    evidence = ppl['evidence'].strip()
    question = ppl['question'].strip()
    # 构建提示：数据库模式 + 问题定义 + 用户问题
    prompt_table = table_info.strip() + '\n\n' + '### definition: ' + evidence + "\n### Question: " + question
    # 调用 LLM 进行前向模式链接
    table_column = gpt(TABLE_AUG_INSTRUCTION, prompt_table)
    table_column = json.loads(table_column)
    return table_column


def preliminary_sql(table_info, table_column, ppl):
    """
    初步 SQL 生成（Preliminary SQL Generation）
    
    功能：基于完整数据库模式生成初步 SQL 查询（SQL_1）
    对应论文：Section III-B2 - Preliminary SQL Generation
    输出：SQL_1 - 基于完整模式生成的 SQL，保留完整数据库结构
    
    处理流程：
    1. 将前向链接结果（table_column）添加到表信息中，作为补充上下文
    2. 构建包含以下内容的提示：
       - 少样本示例（example）：帮助 LLM 理解 SQL 生成格式
       - 完整数据库模式（table_info）：包含表结构、数据样本、外键、前向链接结果
       - 问题定义（evidence）和用户问题（question）
    3. 调用 LLM（使用 SQL_GENERATION_INSTRUCTION）生成 SQL
    4. 解析并清理 SQL 结果（去除换行符）
    
    注意：使用完整模式可以确保数据库结构完整性，降低遗漏必要元素的风险
    
    参数：
        table_info: 完整的数据库表信息
        table_column: 前向模式链接的结果（L_fwd）
        ppl: 包含问题、示例等信息的字典
        
    返回：
        answer: 生成的初步 SQL 查询（SQL_1）
    """
    gpt = model()
    example = ppl['example']
    evidence = ppl['evidence'].strip()
    question = ppl['question'].strip()
    # 将前向链接结果添加到表信息中，作为补充上下文
    table_info += f'### tables: {table_column["tables"]}\n'
    table_info += f'### columns: {table_column["columns"]}\n'

    # 构建完整提示：示例 + 指令 + 数据库模式 + 问题
    table_info = example.strip() + "\n\n### Answer the question by sqlite SQL query only and with no explanation. You must minimize SQL execution time while ensuring correctness.\n" + table_info.strip() + '\n\n### definition: ' + evidence + "\n### Question: " + question

    # 调用 LLM 生成初步 SQL
    answer = gpt(SQL_GENERATION_INSTRUCTION, table_info)
    try:
        answer = json.loads(answer)
    except Exception as e:
        print(e)
        # 处理转义字符问题
        answer = answer.replace("\\", "\\\\")
        answer = json.loads(answer)
    # 清理 SQL：去除换行符，转换为单行
    answer = answer['sql'].replace('\n', ' ')
    return answer


def main(ppl_file, output_file, info_file, x=0):
    """
    Step 1 主函数：处理所有问题，生成初步 SQL 和前向模式链接结果
    
    处理流程：
    1. 加载输入数据（ppl_dev.json）
    2. 对每个问题执行以下步骤：
       a. 构建完整数据库表信息（table_info_construct）
       b. 执行前向模式链接（table_column_selection）→ 得到 L_fwd
       c. 生成初步 SQL（preliminary_sql）→ 得到 SQL_1
       d. 保存结果到文件
    3. 输出文件：
       - output_file (preliminary_sql.txt): 所有问题的初步 SQL（SQL_1）
       - info_file (LLM.json): 前向模式链接结果，供后续后向链接使用
    
    注意：
    - 后向模式链接在 bid_schema_linking.py 中完成，会解析 SQL_1 提取 L_bwd
    - 双向链接结果（L_fwd ∪ L_bwd）用于后续步骤的模式简化
    
    参数：
        ppl_file: 输入文件路径（ppl_dev.json）
        output_file: 输出 SQL 文件路径（preliminary_sql.txt）
        info_file: 输出模式链接信息文件路径（LLM.json）
        x: 起始索引，用于断点续跑
    """
    # 1. 加载输入数据：从 ppl_dev.json 读取所有问题
    with open(ppl_file, 'r') as f:
        ppls = json.load(f)

    answers = []  # 存储所有生成的初步 SQL
    informations = []  # 存储所有前向模式链接结果

    # 2. 遍历处理每个问题
    for i in tqdm(range(x, len(ppls))):
        information = {}
        ppl = ppls[i]

        # 2.1 构建完整数据库表信息
        # 包含：表结构、数据样本、外键信息
        table_info = table_info_construct(ppl)

        # 2.2 前向模式链接（Forward Schema Linking）
        # 功能：使用完整模式，让 LLM 识别与问题相关的表和列
        # 输出：L_fwd（前向链接结果）
        table_column = table_column_selection(table_info, ppl)
        information['tables'] = table_column['tables']
        information['columns'] = table_column['columns']
        informations.append(information)

        # 2.3 生成初步 SQL（Preliminary SQL Generation）
        # 功能：基于完整模式生成 SQL_1
        # 注意：使用完整模式可以保留数据库结构完整性，降低遗漏风险
        pre_sql = preliminary_sql(table_info, table_column, ppl)
        answers.append(pre_sql)

        # 2.4 实时保存结果（防止中断丢失数据）
        # 保存初步 SQL 到文件
        with open(output_file, 'w', encoding='utf-8') as file:
            for sql in answers:
                file.write(str(sql) + '\n')

        # 保存前向模式链接结果到文件（供后续后向链接使用）
        with open(info_file, 'w', encoding='utf-8') as file:
            json.dump(informations, file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()

    # 添加命令行选项

    ## 这里的dataset是ppl_dev.json
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--ppl_file", type=str, default="src/information/ppl_dev.json")
    parser.add_argument("--sql_out_file", type=str, default="src/sql_log/preliminary_sql.txt")
    parser.add_argument("--Schema_linking_LLM", type=str, default="src/schema_linking/LLM.json")
    # 解析命令行参数
    args = parser.parse_args()

    main(args.ppl_file, args.sql_out_file, args.Schema_linking_LLM, args.start_index)
