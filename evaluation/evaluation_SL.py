import json
import copy
import sys
import os

# 添加项目根目录到 Python 路径，以便正确导入 src 模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.util import get_all_schema, extract_tables_and_columns
from src.configs.config import dev_json_path


def recall_get_table(json_file='schema.json'):
    """
    计算模式链接的召回率指标
    
    功能：评估模式链接的严格召回率（SRR）和非严格召回率（NSR）
    对应论文：Section IV-B3, IV-B4 - Strict Recall Rate (SRR) and Non-Strict Recall (NSR)
    
    处理流程：
    1. 从 ground truth SQL 中提取真实的模式元素（表、列）
    2. 从预测的模式链接结果中提取预测的模式元素
    3. 计算严格召回率（SRR）：所有必需元素都被成功召回的比例
    4. 计算非严格召回率（NSR）：召回元素与真实元素的交集比例
    
    参数：
        json_file: 模式链接结果文件路径（默认：schema.json）
    """
    # 存储每个gold sql中，涉及的所有 table_name.column_name
    stats = []
    stats_1 = []
    db_schema_copy = copy.deepcopy(get_all_schema())

    with open(dev_json_path, 'r') as f:
        dev_set = json.load(f)


    # schema linking - 从 ground truth SQL 中提取真实的模式元素
    ground_truths = []
    for example in dev_set:
        ground_truth = []
        ans = extract_tables_and_columns(example['SQL'])
        stats_1.append(len(ans['table']))  # 一个sql语句中涉及的表的数量
        for table in ans['table']:
            for column in ans['column']:
                schema = table + '.' + column
                list_db = [item.lower() for item in db_schema_copy[example['db_id']]]
                if schema.lower() in list_db:
                    ground_truth.append(schema)
        stats.append(len(ground_truth))
        ground_truths.append(ground_truth)

    ### schema linking_pred - 从预测的模式链接结果中提取预测的模式元素
    # 如果 json_file 是相对路径，则基于项目根目录解析
    if not os.path.isabs(json_file):
        json_file = os.path.join(project_root, json_file)
    
    with open(json_file, 'r') as f:
        clms = json.load(f)
    pred_truths = []
    for i in range(len(clms)):
        clm = clms[i]
        pred_truth = []
        columns = clm['columns']
        db_name = dev_set[i]['db_id']
        for column in columns:
            schema = column.replace('`', '')
            if schema.lower() in [item.lower() for item in db_schema_copy[db_name]]:
                pred_truth.append(schema)
        stats.append(len(pred_truth))
        pred_truths.append(pred_truth)

    num = 0
    num_table = 0
    num_column = 0
    num_all = 0
    num_nsr = 0

    t = []
    for ground_truth, pred_truth in zip(ground_truths, pred_truths):
        x1 = set(item.lower() for item in ground_truth)
        x2 = set(item.lower() for item in pred_truth)

        table = set(item.split('.')[0] for item in pred_truth)
        num_table += len(table)
        num_column += len(x2)
        num_all += len(x1)
        num_nsr += len(x1.intersection(x2))

        if x1.issubset(x2):
            num += 1
            t.append(1)
        else:
            t.append(0)

    print("SRR: ", num / len(ground_truths))
    print("Avg.T: ", num_table / len(ground_truths))
    print("Avg.C: ", num_column / len(ground_truths))
    print("NSR: ", num_nsr / num_all)


recall_get_table(json_file='src/schema_linking/schema.json')
