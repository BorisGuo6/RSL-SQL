"""
Bidirectional Schema Linking (BSL) - 双向模式链接

本文件实现论文中的 Step 1 - 后向模式链接（Backward Schema Linking, BSL）
主要功能：
1. 后向模式链接：解析初步 SQL（SQL_1），提取其中使用的表和列（L_bwd）
2. 从问题定义中提取：从 evidence/hint 中提取可能相关的列
3. 合并链接结果：合并前向链接（L_fwd）、后向链接（L_bwd）和 hint 的结果
4. 模式简化：基于合并结果生成简化模式（S'、V'）

对应论文：Section III-B3 - Backward Schema Linking

处理流程：
1. extract_from_sql: 从初步 SQL_1 中提取使用的表和列（后向链接）
2. extract_from_hint: 从问题定义（evidence）中提取相关列
3. merge: 合并前向链接（LLM.json）、后向链接（sql.json）和 hint 的结果
4. filter: 过滤并验证模式元素，确保它们存在于数据库中

输出：
- sql.json: 后向链接结果（从 SQL_1 中提取的表和列）
- hint.json: 从问题定义中提取的表和列
- schema.json: 双向链接的最终结果（L_fwd ∪ L_bwd ∪ hint）
"""

import os
import sqlite3
import json
import copy
import argparse
from configs.config import dev_databases_path, dev_json_path


def get_tables_and_columns(sqlite_db_path):
    """
    从 SQLite 数据库中提取所有表和列
    
    功能：连接数据库，查询所有表名和每表的列名，返回格式化的表.列列表
    用于：构建完整的数据库模式信息，供后续模式链接使用
    
    参数：
        sqlite_db_path: SQLite 数据库文件路径
        
    返回：
        list: 包含所有 "table_name.column_name" 格式的字符串列表
    """
    with sqlite3.connect(sqlite_db_path) as conn:
        cursor = conn.cursor()
        # 查询所有表名
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()

        # 为每个表查询所有列，返回 "table_name.column_name" 格式
        return [
            f"{_table[0]}.{_column[1]}"
            for _table in tables
            for _column in cursor.execute(f"PRAGMA table_info('{_table[0]}');").fetchall()
        ]


def return_db_schema():
    """
    返回所有数据库的完整模式信息
    
    功能：遍历所有数据库目录，提取每个数据库的所有表和列信息
    用于：构建全局数据库模式字典，供模式链接和验证使用
    
    返回：
        dict: 字典，键为数据库名，值为该数据库的所有 "table_name.column_name" 列表
    """
    # 读取所有数据库
    db_base_path = dev_databases_path
    db_schema = {}
    for db_name in os.listdir(db_base_path):
        db_path = os.path.join(db_base_path, db_name, db_name + '.sqlite')
        if os.path.exists(db_path):
            # 提取该数据库的所有表和列
            db_schema[db_name] = get_tables_and_columns(db_path)
    return db_schema


def extract_from_hint(output_path):
    """
    从问题定义（evidence/hint）中提取相关列
    
    功能：分析问题定义文本，通过字符串匹配找出可能相关的列
    对应论文：Section III-B1 - Forward Schema Linking（补充方法）
    输出：从问题定义中提取的表和列
    
    处理流程：
    1. 遍历所有问题，获取每个问题的 evidence（问题定义）
    2. 对于每个数据库的所有列，检查列名是否出现在问题定义中
    3. 如果列名在问题定义中出现，则将该列加入预测结果
    4. 从提取的列中提取表名，格式化输出
    
    注意：这是一种基于字符串匹配的简单方法，用于补充前向和后向链接
    
    参数：
        output_path: 输出文件路径（hint.json）
    """
    db_schema_copy = copy.deepcopy(return_db_schema())
    with open(dev_json_path, 'r') as f:
        dev_set = json.load(f)

    pred_truths = []
    for i in range(len(dev_set)):
        hint = dev_set[i]['evidence']  # 问题定义（evidence）
        db_name = dev_set[i]['db_id']
        pred_truth = []
        list_db = [item.lower() for item in db_schema_copy[db_name]]
        # 遍历数据库中的所有列，检查列名是否出现在问题定义中
        for item in list_db:
            table = item.split('.')[0]
            if table == 'sqlite_sequence':  # 跳过 SQLite 系统表
                continue
            column = item.split('.')[1]
            # 如果列名在问题定义中出现，则加入预测结果
            if column in hint.lower():
                pred_truth.append(item)

        pred_truths.append(pred_truth)

    # 从提取的列中提取表名
    tables = []
    for i in range(len(pred_truths)):
        table = []
        # 格式化列名：添加反引号（用于 SQL 中的特殊字符处理）
        pred_truths[i] = [item.replace('.', '.`') + '`' for item in pred_truths[i]]
        for item in pred_truths[i]:
            t = item.split('.')[0]
            if t not in table:
                table.append(t)
        tables.append(table)

    # 构建输出格式
    answers = []
    for i in range(len(pred_truths)):
        answer = {}
        answer['tables'] = tables[i]
        answer['columns'] = pred_truths[i]
        answers.append(answer)

    # 保存结果到文件
    with open(output_path, 'w') as f:
        json.dump(answers, f, indent=4, ensure_ascii=False)


def extract_from_sql(sql_file, output_file):
    """
    后向模式链接（Backward Schema Linking, BSL）- 核心函数
    
    功能：从初步 SQL（SQL_1）中提取使用的表和列
    对应论文：Section III-B3 - Backward Schema Linking
    输出：L_bwd - 后向链接识别的表和列集合
    
    处理流程：
    1. 读取初步 SQL 文件（preliminary_sql.txt）
    2. 对每个 SQL 查询：
       a. 将 SQL 转换为小写
       b. 遍历数据库中的所有列
       c. 如果列名出现在 SQL 中，则将该列加入预测结果
    3. 从提取的列中提取表名
    4. 格式化并保存结果
    
    核心思想：
    - 如果 SQL_1 是正确的，那么它使用的所有列都是必需的
    - 通过字符串匹配（列名在 SQL 中出现），提取这些列
    - 这种方法可以召回一些前向链接可能遗漏的列
    
    注意：
    - 使用列名精确匹配而非 SQL 解析，可以提高召回率
    - 虽然可能引入一些冗余列，但能确保不遗漏必要元素
    
    参数：
        sql_file: 初步 SQL 文件路径（preliminary_sql.txt）
        output_file: 输出文件路径（sql.json）
    """
    db_schema_copy = copy.deepcopy(return_db_schema())
    with open(dev_json_path, 'r') as f:
        dev_set = json.load(f)

    # 读取初步 SQL 文件
    with open(sql_file, 'r') as f:
        clms = f.readlines()

    pred_truths = []
    for i in range(len(clms)):
        clm = clms[i]  # 初步 SQL（SQL_1）
        db_name = dev_set[i]['db_id']

        pred_truth = []
        sql = clm.lower()  # 转换为小写以便匹配
        list_db = [item.lower() for item in db_schema_copy[db_name]]
        # 遍历数据库中的所有列，检查列名是否出现在 SQL 中
        for item in list_db:
            table = item.split('.')[0]
            if table == 'sqlite_sequence':  # 跳过 SQLite 系统表
                continue
            column = item.split('.')[1]
            # 如果列名在 SQL 中出现，则加入预测结果（后向链接的核心逻辑）
            if column in sql:
                pred_truth.append(item)

        pred_truths.append(pred_truth)

    # 从提取的列中提取表名
    tables = []
    for i in range(len(pred_truths)):
        table = []
        # 格式化列名：添加反引号（用于 SQL 中的特殊字符处理）
        pred_truths[i] = [item.replace('.', '.`') + '`' for item in pred_truths[i]]
        for item in pred_truths[i]:
            t = item.split('.')[0]
            if t not in table:
                table.append(t)
        tables.append(table)

    # 构建输出格式
    answers = []
    for i in range(len(pred_truths)):
        answer = {}
        answer['tables'] = tables[i]
        answer['columns'] = pred_truths[i]
        answers.append(answer)

    # 保存后向链接结果到文件
    with open(output_file, 'w') as f:
        json.dump(answers, f, indent=4, ensure_ascii=False)


def merge(sql_file, LLM_file, hint_file, output_file):
    """
    合并前向链接、后向链接和 hint 的结果
    
    功能：将三种模式链接方法的结果合并，得到 L_fwd ∪ L_bwd ∪ hint
    对应论文：Section III-B4 - Schema Simplification（合并步骤）
    输出：双向模式链接的合并结果
    
    处理流程：
    1. 读取三个输入文件：
       - sql_file: 后向链接结果（从 SQL_1 中提取，L_bwd）
       - LLM_file: 前向链接结果（从 LLM 识别，L_fwd）
       - hint_file: 从问题定义中提取的结果
    2. 对每个问题：
       a. 合并三种方法提取的表和列
       b. 转换为小写并去重（使用 set）
       c. 生成最终的模式链接结果
    3. 保存合并结果
    
    核心目标：通过合并多种方法的结果，最大化召回率（SRR）
    
    参数：
        sql_file: 后向链接结果文件（sql.json）
        LLM_file: 前向链接结果文件（LLM.json）
        hint_file: hint 提取结果文件（hint.json）
        output_file: 输出文件路径（schema.json）
    """
    # 读取后向链接结果（从 SQL_1 中提取）
    with open(sql_file, 'r') as f:
        clms = json.load(f)

    # 读取前向链接结果（从 LLM 识别）
    with open(LLM_file, 'r') as f:
        dev_set = json.load(f)

    # 读取 hint 提取结果（从问题定义中提取）
    with open(hint_file, 'r') as f:
        hint = json.load(f)

    answers = []

    # 合并三种方法的结果：L_fwd ∪ L_bwd ∪ hint
    for x, y, z in zip(clms, dev_set, hint):
        answer = {}

        # 合并表和列：后向链接 + 前向链接 + hint
        tables = y['tables'] + x['tables'] + z['tables']  # 前向 + 后向 + hint
        columns = y['columns'] + x['columns'] + z['columns']  # 前向 + 后向 + hint
        # 转换为小写以便统一比较
        tables = [item.lower() for item in tables]
        columns = [item.lower() for item in columns]

        # 去重：使用 set 去除重复的表和列
        tables = list(set(tables))
        columns = list(set(columns))

        answer['tables'] = tables
        answer['columns'] = columns
        answers.append(answer)

    # 保存合并结果
    with open(output_file, 'w') as f:
        json.dump(answers, f, indent=4, ensure_ascii=False)


def filter(dev_file, schema_file, output_file):
    """
    过滤并验证模式元素
    
    功能：验证合并后的模式链接结果，确保所有元素都真实存在于数据库中
    对应论文：Section III-B4 - Schema Simplification（验证步骤）
    输出：经过验证和过滤的最终模式链接结果
    
    处理流程：
    1. 读取合并后的模式链接结果
    2. 对每个问题：
       a. 获取该问题对应的数据库模式
       b. 检查模式链接结果中的每个列是否真实存在于数据库中
       c. 只保留存在于数据库中的列
    3. 从验证后的列中提取表名
    4. 格式化并保存最终结果
    
    核心目标：确保模式链接结果中的所有元素都是有效的数据库元素
    
    参数：
        dev_file: 开发集文件路径（dev.json）
        schema_file: 合并后的模式链接结果文件（schema.json，也是输出文件）
        output_file: 输出文件路径（schema.json，覆盖输入文件）
    """
    db_schema_copy = copy.deepcopy(return_db_schema())

    with open(dev_file, 'r') as f:
        dev_set = json.load(f)

    with open(schema_file, 'r') as f:
        informations = json.load(f)

    preds = []
    for i in range(len(informations)):
        pred = []

        information = informations[i]  # 合并后的模式链接结果
        db = dev_set[i]['db_id']

        # 获取该数据库的真实模式
        db_schema = db_schema_copy[db]
        # 清理格式（去除反引号）
        db_schema = [obj.replace('`', '') for obj in db_schema]

        columns = information['columns']
        columns = [obj.replace('`', '').lower() for obj in columns]

        # 验证：只保留真实存在于数据库中的列
        for obj in db_schema:
            if obj.lower() in columns and obj.lower() not in pred:
                pred.append(obj)

        preds.append(pred)

    # 从验证后的列中提取表名
    tables = []
    for i in range(len(preds)):
        table = []
        # 格式化列名：添加反引号
        preds[i] = [item.replace('.', '.`') + '`' for item in preds[i]]
        for item in preds[i]:
            t = item.split('.')[0]
            if t not in table:
                table.append(t)
        tables.append(table)

    # 构建最终输出格式
    answers = []
    for i in range(len(preds)):
        answer = {}
        answer['tables'] = tables[i]
        answer['columns'] = preds[i]
        answers.append(answer)

    # 保存验证后的最终结果（覆盖输入文件）
    with open(output_file, 'w') as f:
        json.dump(answers, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    """
    双向模式链接主函数：完成后向链接并合并所有结果
    
    处理流程：
    1. 后向模式链接（extract_from_sql）：
       - 从初步 SQL_1 中提取使用的表和列（L_bwd）
       - 输出：sql.json
    2. 从问题定义中提取（extract_from_hint）：
       - 从 evidence/hint 中提取相关列
       - 输出：hint.json
    3. 合并链接结果（merge）：
       - 合并前向链接（L_fwd）、后向链接（L_bwd）和 hint 的结果
       - 得到 L_fwd ∪ L_bwd ∪ hint
       - 输出：schema.json（临时）
    4. 过滤验证（filter）：
       - 验证合并结果中的元素是否真实存在于数据库中
       - 只保留有效的数据库元素
       - 输出：schema.json（最终）
    
    核心目标：通过双向链接最大化召回率，达到严格召回率 94%
    
    输入文件：
        - preliminary_sql.txt: Step 1 生成的初步 SQL（SQL_1）
        - LLM.json: Step 1 生成的前向链接结果（L_fwd）
    
    输出文件：
        - sql.json: 后向链接结果（L_bwd）
        - hint.json: 从问题定义中提取的结果
        - schema.json: 双向链接的最终结果（L_fwd ∪ L_bwd ∪ hint，已验证）
    """
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()

    # 添加命令行选项
    parser.add_argument("--pre_sql_file", type=str, default="src/sql_log/preliminary_sql.txt")
    parser.add_argument("--sql_sl_output", type=str, default="src/schema_linking/sql.json")
    parser.add_argument("--hint_sl_output", type=str, default="src/schema_linking/hint.json")
    parser.add_argument("--LLM_sl_output", type=str, default="src/schema_linking/LLM.json")
    parser.add_argument("--Schema_linking_output", type=str, default="src/schema_linking/schema.json")
    
    # 解析命令行参数
    args = parser.parse_args()

    # Step 1: 后向模式链接 - 从初步 SQL_1 中提取使用的表和列（L_bwd）
    extract_from_sql(args.pre_sql_file, args.sql_sl_output)
    
    # Step 2: 从问题定义中提取相关列
    extract_from_hint(args.hint_sl_output)
    
    # Step 3: 合并链接结果 - 合并前向链接、后向链接和 hint 的结果
    # 得到 L_fwd ∪ L_bwd ∪ hint
    merge(args.sql_sl_output, args.LLM_sl_output, args.hint_sl_output, args.Schema_linking_output)
    
    # Step 4: 过滤验证 - 确保所有元素都真实存在于数据库中
    filter(dev_json_path, args.Schema_linking_output, args.Schema_linking_output)
