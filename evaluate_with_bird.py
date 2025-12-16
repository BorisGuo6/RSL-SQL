#!/usr/bin/env python3
"""
使用 BIRD benchmark 评估 RSL-SQL 的结果
用法: python evaluate_with_bird.py --sql_file src/sql_log/final_sql.txt
"""

import json
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
import sys
import os

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def convert_txt_to_bird_format(sql_txt_path, dev_json_path, output_path):
    """将 RSL-SQL 的 txt 输出转换为 BIRD 评估格式"""
    with open(sql_txt_path, 'r') as f:
        sqls = f.readlines()
    
    dev_data = load_json(dev_json_path)
    
    result = {}
    for i, sql in enumerate(sqls):
        sql = sql.strip()
        if i < len(dev_data):
            db_id = dev_data[i]['db_id']
            # 添加分号（如果没有）并转换为 BIRD 格式
            if not sql.endswith(';'):
                sql = sql + ';'
            result[str(i)] = f"{sql}\t----- bird -----\t{db_id}"
        else:
            print(f"Warning: SQL index {i} exceeds dev.json length {len(dev_data)}")
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    print(f"Converted {len(result)} SQL queries to BIRD format: {output_path}")
    return result

exec_result = []

def result_callback(result):
    exec_result.append(result)

def execute_sql(predicted_sql, ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    conn.close()
    return res

def execute_model(predicted_sql, ground_truth, db_place, idx, meta_time_out):
    try:
        res = func_timeout(meta_time_out, execute_sql,
                          args=(predicted_sql, ground_truth, db_place))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        res = 0
    except Exception as e:
        res = 0
    return {'sql_idx': idx, 'res': res}

def package_sqls_from_json(json_path, db_root_path):
    """从 BIRD 格式的 JSON 文件中提取 SQL"""
    clean_sqls = []
    db_path_list = []
    
    sql_data = load_json(json_path)
    for idx in range(len(sql_data)):
        sql_str = sql_data[str(idx)]
        if isinstance(sql_str, str) and '\t----- bird -----\t' in sql_str:
            sql, db_name = sql_str.split('\t----- bird -----\t')
            sql = sql.rstrip(';').strip()
        else:
            sql, db_name = " ", "financial"
        clean_sqls.append(sql)
        db_path_list.append(os.path.join(db_root_path, db_name, f"{db_name}.sqlite"))
    
    return clean_sqls, db_path_list

def package_sqls_from_gold(gold_path, db_root_path):
    """从 gold.sql 文件中提取 SQL"""
    clean_sqls = []
    db_path_list = []
    
    with open(gold_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                sql, db_name = line.split('\t')
                clean_sqls.append(sql)
                db_path_list.append(os.path.join(db_root_path, db_name, f"{db_name}.sqlite"))
    
    return clean_sqls, db_path_list

def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0):
    global exec_result
    exec_result = []
    
    pool = mp.Pool(processes=num_cpus)
    for i, sql_pair in enumerate(sqls):
        predicted_sql, ground_truth = sql_pair
        pool.apply_async(execute_model, 
                        args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out), 
                        callback=result_callback)
    pool.close()
    pool.join()
    return exec_result

def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_acc_by_diff(exec_results, diff_json_path):
    num_queries = len(exec_results)
    results = [res['res'] for res in exec_results]
    contents = load_json(diff_json_path)
    
    simple_results, moderate_results, challenging_results = [], [], []
    
    for i, content in enumerate(contents):
        if i < len(exec_results):
            if content['difficulty'] == 'simple':
                simple_results.append(exec_results[i])
            elif content['difficulty'] == 'moderate':
                moderate_results.append(exec_results[i])
            elif content['difficulty'] == 'challenging':
                challenging_results.append(exec_results[i])
    
    simple_acc = sum([res['res'] for res in simple_results]) / len(simple_results) if simple_results else 0
    moderate_acc = sum([res['res'] for res in moderate_results]) / len(moderate_results) if moderate_results else 0
    challenging_acc = sum([res['res'] for res in challenging_results]) / len(challenging_results) if challenging_results else 0
    all_acc = sum(results) / num_queries if num_queries else 0
    
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, all_acc * 100, count_lists

def print_results(score_lists, count_lists):
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("\n" + "=" * 90)
    print("                           RSL-SQL BIRD Benchmark Evaluation")
    print("=" * 90)
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))
    print("-" * 90)
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy (%)', *score_lists))
    print("=" * 90)

def main():
    parser = argparse.ArgumentParser(description='Evaluate RSL-SQL results with BIRD benchmark')
    parser.add_argument('--sql_file', type=str, default='src/sql_log/final_sql.txt',
                       help='Path to RSL-SQL output file (default: src/sql_log/final_sql.txt)')
    parser.add_argument('--dev_json', type=str, default='data/dev.json',
                       help='Path to dev.json file')
    parser.add_argument('--gold_sql', type=str, default='external/damo-bird-llm/data/dev_gold.sql',
                       help='Path to ground truth SQL file')
    parser.add_argument('--db_root', type=str, default='database/dev_databases',
                       help='Path to database root directory')
    parser.add_argument('--num_cpus', type=int, default=4,
                       help='Number of CPUs for parallel execution')
    parser.add_argument('--timeout', type=float, default=30.0,
                       help='Timeout for each SQL execution')
    parser.add_argument('--output_json', type=str, default='src/sql_log/predict_dev.json',
                       help='Output path for converted JSON file')
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 转换为绝对路径
    sql_file = os.path.join(script_dir, args.sql_file) if not os.path.isabs(args.sql_file) else args.sql_file
    dev_json = os.path.join(script_dir, args.dev_json) if not os.path.isabs(args.dev_json) else args.dev_json
    gold_sql = os.path.join(script_dir, args.gold_sql) if not os.path.isabs(args.gold_sql) else args.gold_sql
    db_root = os.path.join(script_dir, args.db_root) if not os.path.isabs(args.db_root) else args.db_root
    output_json = os.path.join(script_dir, args.output_json) if not os.path.isabs(args.output_json) else args.output_json
    
    print(f"SQL File: {sql_file}")
    print(f"Dev JSON: {dev_json}")
    print(f"Gold SQL: {gold_sql}")
    print(f"DB Root: {db_root}")
    
    # Step 1: 转换格式
    print("\n[Step 1] Converting RSL-SQL output to BIRD format...")
    convert_txt_to_bird_format(sql_file, dev_json, output_json)
    
    # Step 2: 加载预测和 ground truth
    print("\n[Step 2] Loading predicted and ground truth SQL...")
    pred_sqls, pred_db_paths = package_sqls_from_json(output_json, db_root)
    gt_sqls, gt_db_paths = package_sqls_from_gold(gold_sql, db_root)
    
    print(f"  Predicted SQLs: {len(pred_sqls)}")
    print(f"  Ground Truth SQLs: {len(gt_sqls)}")
    
    # 确保数量匹配
    min_len = min(len(pred_sqls), len(gt_sqls))
    if len(pred_sqls) != len(gt_sqls):
        print(f"  Warning: Length mismatch! Using first {min_len} queries.")
        pred_sqls = pred_sqls[:min_len]
        gt_sqls = gt_sqls[:min_len]
        pred_db_paths = pred_db_paths[:min_len]
    
    # Step 3: 执行评估
    print(f"\n[Step 3] Running evaluation with {args.num_cpus} CPUs (timeout: {args.timeout}s)...")
    query_pairs = list(zip(pred_sqls, gt_sqls))
    results = run_sqls_parallel(query_pairs, pred_db_paths, 
                                num_cpus=args.num_cpus, 
                                meta_time_out=args.timeout)
    results = sort_results(results)
    
    # Step 4: 计算准确率
    print("\n[Step 4] Computing accuracy by difficulty...")
    simple_acc, moderate_acc, challenging_acc, total_acc, count_lists = \
        compute_acc_by_diff(results, dev_json)
    
    score_lists = [simple_acc, moderate_acc, challenging_acc, total_acc]
    print_results(score_lists, count_lists)
    
    # 统计正确数量
    correct = sum([r['res'] for r in results])
    print(f"\nCorrect: {correct}/{len(results)}")
    print("Evaluation completed!")

if __name__ == '__main__':
    main()

