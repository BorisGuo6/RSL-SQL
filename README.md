<div align="center">
  <h1><a href="https://arxiv.org/abs/2411.00073">RSL-SQL: Robust Schema Linking in Text-to-SQL Generation</a></h1>
</div>


<h5 align="center"> Please give us a star ⭐ for the latest update.  </h5>

<h5 align="center">

 
[![arXiv](https://img.shields.io/badge/Arxiv-2411.00073-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.00073) 
  <br>
</h5>

## Overview

![](figs/framework.jpg)

### 框架说明

RSL-SQL 是一个基于鲁棒模式链接的 Text-to-SQL 生成框架，通过四个核心步骤最大化模式链接的收益并降低其风险：

**Step 1: 双向模式链接（Bidirectional Schema Linking, BSL）**
- **前向模式链接**：使用完整数据库模式，让 LLM 识别与问题相关的表和列（L_fwd）
- **初步 SQL 生成**：基于完整模式生成 SQL_1，保留完整数据库结构，降低遗漏风险
- **后向模式链接**：解析 SQL_1，提取其中使用的表和列（L_bwd），通过合并 L_fwd ∪ L_bwd 提升召回率
- **模式简化**：基于双向链接结果生成简化模式，在保持高召回率（严格召回率 94%）的同时减少输入列数 83%

**Step 2: 上下文信息增强（Contextual Information Augmentation, CIA）**
- **模式简化**：基于 Step 1 的链接结果生成简化模式 S'、V'、D'
- **SQL 组件生成**：
  - 生成需要的表/列元素（H_E）
  - 生成 SQL 关键词如 DISTINCT、GROUP BY（H_K）
  - 生成 WHERE 条件（H_C）
- **上下文增强**：合并列描述 D' 与生成的组件信息，帮助 LLM 更好理解简化模式
- **SQL 生成**：基于简化模式和增强信息生成 SQL_2，提升正收益（y），减少模式简化带来的信息损失

**Step 3: 二元选择策略（Binary Selection Strategy, BSS）**
- **执行比较**：执行 SQL_1（完整模式）和 SQL_2（简化模式），得到执行结果 R_1、R_2
- **最优选择**：使用 LLM 比较两个 SQL 及其执行结果，选择更优的作为 SQL_3
- **风险对冲**：完整模式保留结构但可能有冗余，简化模式减少噪声但可能遗漏信息，通过选择策略平衡两者，降低负影响（x）同时提升正收益（y）

**Step 4: 多轮自我纠正（Multi-Turn Self-Correction, MTSC）**
- **错误检测**：执行 SQL_3，检查语法错误或空结果
- **迭代纠正**：如果执行失败或返回空结果，将错误信息反馈给 LLM，生成修正后的 SQL_4，最多进行 5 轮迭代
- **终止条件**：SQL 执行成功且返回非空结果，或达到最大迭代轮数

**核心目标**：最大化模式链接的正收益（y），最小化模式链接的负影响（x），在 BIRD 数据集上达到 67.21% 执行准确率（开源 SOTA）

### 整体流程

```
完整数据库模式 (S, V, D)
    ↓
[Step 1: 双向模式链接]
    ├─ 前向模式链接 → L_fwd
    ├─ 初步 SQL 生成 → SQL_1
    ├─ 后向模式链接 → L_bwd
    └─ 模式简化 → S', V', D'
    ↓
简化模式 + 上下文增强信息
    ↓
[Step 2: 上下文信息增强]
    ├─ SQL 组件生成 (H_E, H_C, H_K)
    ├─ 上下文增强 (H_Aug)
    └─ SQL 生成 → SQL_2
    ↓
SQL_1 (完整模式) + SQL_2 (简化模式)
    ↓
[Step 3: 二元选择策略]
    ├─ 执行 SQL_1 → R_1
    ├─ 执行 SQL_2 → R_2
    └─ LLM 选择最优 → SQL_3
    ↓
[Step 4: 多轮自我纠正]
    ├─ 执行 SQL_3
    ├─ 检测错误/空结果
    ├─ 迭代纠正 (最多 5 轮)
    └─ 最终 SQL → SQL_4
```

## Main Results

### Execution Accuracy on BIRD Dev Set
![](figs/main_bird.png)

## Ablaition Study

![](figs/ablation.png)



## Project directory structure

- Download `pytorch_model.bin` and place it in the `few_shot/sentence_transformers/` folder. Download address: https://huggingface.co/sentence-transformers/all-mpnet-base-v2/tree/main

- Download the `column_meaning.json` file and place it in the `data/` folder. Download address: https://github.com/quge2023/TA-SQL/blob/master/outputs/column_meaning.json

- Download the `dev.json` file and `dev_tables.json` file of the development set in the `data/` folder. Download address: https://bird-bench.github.io/

- Download the `train-00000-of-00001-fe8894d41b7815be.parquet` file and place it in the `few_shot/` folder. Download address: https://huggingface.co/datasets/xu3kev/BIRD-SQL-data-train/tree/main/data
- Download the dev_databases file of the development set in the database/ dev_databases. Download address: https://bird-bench.github.io/





```plaintext
RSL-SQL/
├── README.md
├── requirements.txt
│
├── data/
│   ├── column_meaning.json
│   ├── dev.json
│   └── dev_tables.json
│
├── database/
│   └── dev_databases/
│
├── few_shot/
│   ├── sentence_transformers/
│   └── train-00000-of-00001-fe8894d41b7815be.parquet
│
└── src/
    └── configs/
        └── config.py
```

## environment



```bash
conda create -n rsl_sql python=3.10
conda activate rsl_sql
pip install -r requirements.txt
```
### 配置说明

修改 `src/configs/config.py` 中的参数配置：

```python
import os

# 数据库路径：开发集数据库文件夹路径
dev_databases_path = 'database/dev_databases'

# 数据文件路径：开发集 JSON 文件路径
dev_json_path = 'data/dev.json'

# API 配置：从环境变量读取，如果没有设置则使用默认值
# 设置方式：export OPENAI_API_KEY="your_api_key"
api = os.getenv('OPENAI_API_KEY', '')

# 模型配置：从环境变量读取，推荐使用 'o4'（最强模型）或 'gpt-4o'
# 设置方式：export OPENAI_MODEL="o4"
model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

# Base URL：从环境变量读取，如果使用官方 API 可省略
# 设置方式：export OPENAI_BASE_URL="https://api.openai.com/v1"
base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
```

**环境变量设置示例**（在终端中执行或添加到 `~/.zshrc`）：

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="o4"  # 或 "gpt-4o"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选，如果使用官方 API
```







## RUN

### 1. Data Preprocessing（数据预处理）

**作用**：准备输入数据，包括构建问题-数据库对、生成少样本示例等

```bash
# 构建 ppl_dev.json：将原始 dev.json 和 dev_tables.json 整合，添加数据库结构信息（表结构、外键、数据样本等）
python src/data_construct.py 

# 构建少样本示例对：从训练集中提取问题-SQL 对，用于后续的相似度匹配
python few_shot/construct_QA.py 

# 生成少样本示例：使用 sentence-transformers 模型计算问题相似度，为每个问题选择 top-k 个最相似的示例
# --kshot 3 表示每个问题选择 3 个示例
python few_shot/slg_main.py --dataset src/information/ppl_dev.json --out_file src/information/example.json --kshot 3

# 将生成的少样本示例添加到 ppl_dev.json 中，供后续步骤使用
python src/information/add_example.py
```



### 2. Preliminary SQL Generation and Bidirectional Schema Linking（初步 SQL 生成与双向模式链接）

**作用**：实现 Step 1 - 双向模式链接（BSL），生成初步 SQL 并完成模式简化

```bash
# Step 1: 初步 SQL 生成
# 功能：
#   1. 前向模式链接：使用完整数据库模式，让 LLM 识别与问题相关的表和列（L_fwd）
#   2. 初步 SQL 生成：基于完整模式生成 SQL_1，保留完整数据库结构，降低遗漏风险
#   3. 保存模式链接结果：将前向链接的结果保存到 LLM.json
# 输出文件：
#   - preliminary_sql.txt: 每个问题对应的初步 SQL（SQL_1）
#   - LLM.json: 前向模式链接的结果（包含表、列信息）
# 注意：如果运行中断，需要及时保存这两个文件，以便后续继续运行
python src/step_1_preliminary_sql.py --ppl_file src/information/ppl_dev.json --sql_out_file src/sql_log/preliminary_sql.txt --Schema_linking_LLM src/schema_linking/LLM.json --start_index 0

# 双向模式链接：完成后向模式链接并合并结果
# 功能：
#   1. 后向模式链接：解析 preliminary_sql.txt 中的 SQL_1，提取使用的表和列（L_bwd）
#   2. 合并链接结果：合并前向（L_fwd）和后向（L_bwd）链接结果，得到 L_fwd ∪ L_bwd
#   3. 生成简化模式：基于合并结果生成简化后的数据库模式（S'、V'）
# 输出文件：
#   - schema.json: 双向模式链接的最终结果（包含需要保留的表和列）
python src/bid_schema_linking.py --pre_sql_file src/sql_log/preliminary_sql.txt --sql_sl_output src/schema_linking/sql.json --hint_sl_output src/schema_linking/hint.json --LLM_sl_output src/schema_linking/LLM.json --Schema_linking_output src/schema_linking/schema.json
cp src/schema_linking/schema.json src/information

# 将模式链接结果添加到 ppl_dev.json 中，供后续步骤使用
python src/information/add_sl.py
```

### 3. SQL Generation based Simplified Schema and Information Augmentation（基于简化模式的 SQL 生成与信息增强）

**作用**：实现 Step 2 - 上下文信息增强（CIA），基于简化模式生成增强的 SQL

```bash
# Step 2: 基于简化模式的 SQL 生成与上下文信息增强
# 功能：
#   1. 模式简化：基于 Step 1 的模式链接结果，生成简化后的数据库模式（S'、V'、D'）
#   2. SQL 组件生成：
#      - 表/列元素生成（H_E）：识别需要的表和列
#      - SQL 关键词生成（H_K）：生成如 DISTINCT、GROUP BY 等关键词
#      - 条件生成（H_C）：从问题中分解出 WHERE 条件
#   3. 上下文增强：合并列描述 D' 与生成的组件信息（H_Aug = {D', H_E, H_C, H_K}）
#   4. SQL 生成：基于简化模式和增强信息生成 SQL_2
# 输出文件：
#   - step_2_information_augmentation.txt: 每个问题对应的增强 SQL（SQL_2）
#   - augmentation.json: 生成的上下文增强信息（包含表、列、关键词、条件等）
# 注意：如果运行中断，需要及时保存这两个文件，以便后续继续运行
python src/step_2_information_augmentation.py --ppl_file src/information/ppl_dev.json --sql_2_output src/sql_log/step_2_information_augmentation.txt --information_output src/information/augmentation.json --start_index 0

# 将上下文增强信息添加到 ppl_dev.json 中，供后续步骤使用
python src/information/add_augmentation.py
```

### 4. Binary Selection Strategy（二元选择策略）

**作用**：实现 Step 3 - 二元选择策略（BSS），从两个 SQL 中选择最优的

```bash
# Step 3: 二元选择策略
# 功能：
#   1. 执行两个 SQL：分别执行 SQL_1（完整模式）和 SQL_2（简化模式），得到执行结果 R_1、R_2
#   2. 最优选择：使用 LLM 比较两个 SQL 及其执行结果，分析哪个更符合问题语义
#   3. 风险对冲：通过选择策略平衡完整模式（保留结构但可能有冗余）和简化模式（减少噪声但可能遗漏信息）的优缺点
#   4. 生成 SQL_3：选择更优的 SQL 作为 SQL_3
# 输入文件：
#   - preliminary_sql.txt: Step 1 生成的 SQL_1
#   - step_2_information_augmentation.txt: Step 2 生成的 SQL_2
# 输出文件：
#   - step_3_binary.txt: 每个问题对应的最优 SQL（SQL_3）
# 注意：如果运行中断，需要及时保存输出文件，以便后续继续运行
python src/step_3_binary_selection.py --ppl_file src/information/ppl_dev.json --sql_3_output src/sql_log/step_3_binary.txt --sql_1 src/sql_log/preliminary_sql.txt --sql_2 src/sql_log/step_2_information_augmentation.txt --start_index 0
```

### 5. Multi-Turn Self-Correction（多轮自我纠正）

**作用**：实现 Step 4 - 多轮自我纠正（MTSC），通过执行反馈迭代优化错误 SQL

```bash
# Step 4: 多轮自我纠正
# 功能：
#   1. 错误检测：执行 SQL_3，检查是否有语法错误或返回空结果
#   2. 迭代纠正：
#      - 如果执行失败或返回空结果，将错误信息 E(i) 反馈给 LLM
#      - LLM 基于错误信息生成修正后的 SQL_4(i+1)
#      - 最多进行 5 轮迭代（num < 5）
#   3. 终止条件：SQL 执行成功且返回非空结果，或达到最大迭代轮数
# 输入文件：
#   - step_3_binary.txt: Step 3 选择的最优 SQL（SQL_3）
# 输出文件：
#   - final_sql.txt: 每个问题对应的最终优化后的 SQL（SQL_4）
python src/step_4_self_correction.py --ppl_file src/information/ppl_dev.json --sql_4_output src/sql_log/final_sql.txt --sql_refinement src/sql_log/step_3_binary.txt --start_index 0
```

## Evaluation 
### Execution (EX) Evaluation:
Refer to the official evaluation script, the link is: https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird

### Strict Recall Rate Evaluation:

The script is in the `evaluation/evaluation_SL.py` file, and the usage is as follows:
We should organize the output of the database elements in the following format:
```json
{
        "tables": [
            "frpm"
        ],
        "columns": [
            "frpm.`Free Meal Count (K-12)`",
            "frpm.`Enrollment (K-12)`",
            "frpm.`School Name`",
            "frpm.`County Name`"
        ]
    }
```




# Citation
```citation
@article{cao2024rsl,
  title={RSL-SQL: Robust Schema Linking in Text-to-SQL Generation},
  author={Cao, Zhenbiao and Zheng, Yuanlei and Fan, Zhihao and Zhang, Xiaojin and Chen, Wei},
  journal={arXiv preprint arXiv:2411.00073},
  year={2024}
}
```
