"""
Step 4: Multi-Turn Self-Correction (MTSC)
多轮自我纠正

本步骤实现论文中的 Step 4 - 多轮自我纠正（Multi-Turn Self-Correction, MTSC）
主要功能：
1. 错误检测：执行 SQL_3，检查是否有语法错误或返回空结果
2. 迭代纠正：
   - 如果执行失败或返回空结果，将错误信息 E(i) 反馈给 LLM
   - LLM 基于错误信息生成修正后的 SQL_4(i+1)
   - 最多进行 5 轮迭代（num < 5）
3. 终止条件：SQL 执行成功且返回非空结果，或达到最大迭代轮数

输出：
- final_sql.txt: 每个问题对应的最终优化后的 SQL（SQL_4）

核心目标：通过执行反馈迭代优化错误 SQL，进一步提升准确率
"""

from llm.self_correction_gpt import GPT
import json
from tqdm import tqdm
from utils.util import execute_sql
from configs.Instruction import SELF_CORRECTION_PROMPT
from utils.simplified_schema import simplified, explanation_collection
import argparse


def table_info_construct(ppl, simple_ddl, ddl_data, foreign_key, explanation):
    """
    构建用于自我纠正的表信息提示
    
    功能：构建包含简化模式、上下文增强信息、问题等的提示，用于多轮自我纠正
    对应论文：Section III-E - Multi-Turn Self-Correction
    
    与 Step 2 的区别：
    - 添加了 SQL 关键词和条件信息（从 ppl 中获取，由 Step 2 生成）
    - 使用 "Hint" 而非 "definition"，强调提示的作用
    
    参数：
        ppl: 包含问题、SQL 关键词、条件等信息的字典
        simple_ddl: 简化后的表结构
        ddl_data: 简化后的数据样本
        foreign_key: 外键信息
        explanation: 列描述信息
        
    返回：
        table_info: 格式化后的表信息字符串，用于自我纠正
    """
    question = ppl['question'].strip()
    evidence = ppl['evidence'].strip()
    example = ppl['example']

    # 构建简化模式信息（包含列描述）
    table_info = '### Sqlite SQL tables, with their properties:\n'
    table_info += simple_ddl + '\n' + '### Here are some data information about database references.\n' + ddl_data + '\n### Foreign key information of Sqlite SQL tables, used for table joins:\n' + foreign_key + '\n### The meaning of every column:\n#\n' + explanation.strip() + "\n#\n"

    # 添加上下文增强信息（由 Step 2 生成）
    table_info += f'\n### sql_keywords: {ppl["sql_keywords"]}'  # H_K
    table_info += f'\n### conditions: {ppl["conditions"]}'  # H_C

    # 构建完整提示：示例 + 指令 + 简化模式 + 增强信息 + 问题提示
    table_info = example.strip() + '\n\n' + "### Answer the question by sqlite SQL query only and with no explanation. You must minimize SQL execution time while ensuring correctness.\n" + table_info.strip() + '\n\n' + '### Hint: ' + evidence + "\n### Question: " + question + '\n\n' + 'The hint aims to direct your focus towards the specific elements of the database schema that are crucial for answering the question effectively.'

    return table_info


def main(ppl_file, sql_file, output_file, x=0):
    """
    Step 4 主函数：处理所有问题，通过多轮自我纠正优化 SQL
    
    处理流程：
    1. 加载输入数据：
       - ppl_dev.json：包含问题、数据库、上下文增强信息等
       - step_3_binary.txt：Step 3 选择的最优 SQL（SQL_3）
    2. 对每个问题执行多轮自我纠正：
       a. 模式简化（simplified）→ 生成简化模式 S'、V'
       b. 列描述收集（explanation_collection）→ 生成列描述 D'
       c. 构建表信息（table_info_construct）→ 包含简化模式、增强信息、问题
       d. 迭代纠正循环（最多 5 轮）：
          - 执行当前 SQL，检查是否有错误或空结果
          - 如果 row_count == 0 且 column_count == 0（执行失败或空结果）：
            * 将错误信息反馈给 LLM
            * LLM 生成修正后的 SQL
            * 继续下一轮迭代
          - 如果执行成功且返回非空结果，退出循环
       e. 保存最终 SQL 到文件
    3. 输出文件：
       - output_file (final_sql.txt): 所有问题的最终优化 SQL（SQL_4）
    
    核心目标：通过执行反馈迭代优化错误 SQL，进一步提升准确率
    
    参数：
        ppl_file: 输入文件路径（ppl_dev.json，已包含上下文增强信息）
        sql_file: Step 3 的 SQL 文件路径（step_3_binary.txt）
        output_file: 输出 SQL 文件路径（final_sql.txt）
        x: 起始索引，用于断点续跑
    """
    gpt = GPT()

    # 1. 加载输入数据
    with open(ppl_file, 'r') as f:
        ppls = json.load(f)

    # 加载 Step 3 选择的最优 SQL（SQL_3）
    with open(sql_file, 'r') as f:
        pre_sqls = f.readlines()

    sys_message = SELF_CORRECTION_PROMPT  # 自我纠正的系统提示

    answers = []  # 存储所有最终优化的 SQL

    # 2. 遍历处理每个问题
    for i in tqdm(range(x, len(ppls))):
        message = []  # 多轮对话消息列表
        message.append({'role': 'system', 'content': sys_message})
        ppl = ppls[i]
        db = ppl['db']

        # 2.1 模式简化：生成简化模式 S'、V'
        simple_ddl, ddl_data, foreign_key = simplified(ppl)

        # 2.2 列描述收集：生成列描述 D'
        explanation = explanation_collection(ppl)

        # 2.3 构建表信息：包含简化模式、上下文增强信息、问题
        table_info = table_info_construct(ppl, simple_ddl, ddl_data, foreign_key, explanation)

        pre_sql = pre_sqls[i].strip()  # 初始 SQL（SQL_3）

        # 2.4 多轮自我纠正循环（最多 5 轮）
        num = 0
        while num < 5:
            # 执行当前 SQL，检查是否有错误或空结果
            row_count, column_count, result = execute_sql(pre_sql, db)

            # 如果不是第一轮，构建包含错误信息的提示
            if num > 0:
                table_info = "### Buggy SQL: " + pre_sql.strip() + "\n" + f"### The result of the buggy SQL is [{result.strip()}]. Please fix the SQL to get the correct result."
            
            # 如果执行失败或返回空结果（row_count == 0 且 column_count == 0）
            if row_count == 0 and column_count == 0:
                # 将错误信息添加到对话中
                message.append({'role': 'user', 'content': table_info})
                # 调用 LLM 生成修正后的 SQL
                message, answer = gpt(message)
                num += 1
                try:
                    answer = json.loads(answer)
                except Exception as e:
                    # 处理转义字符问题
                    answer = answer.replace('\\', '\\\\')
                    try:
                        answer = json.loads(answer)
                    except Exception as e:
                        break  # 解析失败，退出循环
                # 更新 SQL 为修正后的版本
                pre_sql = answer['sql'].strip()
            else:
                # 执行成功且返回非空结果，退出循环
                break
        
        # 2.5 保存最终 SQL（清理换行符）
        answers.append(pre_sql.replace('\n', ' '))
        # 实时保存结果（防止中断丢失数据）
        with open(output_file, 'w') as f:
            for answer in answers:
                f.write(answer + '\n')


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()

    # 添加命令行选项

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--ppl_file", type=str, default="src/information/ppl_dev.json")
    parser.add_argument("--sql_4_output", type=str, default="src/sql_log/final_sql.txt")
    parser.add_argument("--sql_refinement", type=str, default="src/sql_log/step_3_binary.txt")

    # 解析命令行参数
    args = parser.parse_args()

    main(args.ppl_file, args.sql_refinement, args.sql_4_output, args.start_index)
