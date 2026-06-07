# AutoSolver Agent

AutoSolver Agent是一个面向任务分配问题的离线LLM自动求解器实验项目。项目把本地候选数据、算法组件和模型反馈组织成多轮生成流程，让LLM不断重写、组合和改进独立的`solver.py`，再通过本地评测选择当前最优版本。

## 项目简介

本项目的核心目标是自动生成可提交的任务分配求解器。输入数据由任务组、骑手、总成本分数和意愿值组成，系统会要求生成的求解器尽量覆盖更多任务，并在覆盖完整的前提下降低综合成本。整体流程包含读取案例、选择算法素材、调用DeepSeek/OpenAI兼容接口生成候选代码、本地运行候选代码、记录每轮结果并输出最佳求解器。

项目同时提供一个Streamlit Dashboard，用于配置案例、模型参数、运行轮数和评测环境，并实时查看运行日志、候选解和迭代曲线。

## 核心特性

- 多轮LLM生成：支持按轮生成、重组和精修候选求解器。
- 本地自动评测：对生成代码进行语法检查、超时控制和案例评估。
- 算法组件注入：从`algorithm_components`中选择解析、目标函数、初始构造、匹配、局部搜索等素材提供给模型。
- 自适应素材释放：支持`adaptive`、`staged`和`open`三种组件释放策略。
- 候选对决机制：支持精修候选和重组候选的对比评估。
- 可视化控制台：通过Streamlit展示运行状态、历史记录和最终结果。

## 目录结构

```text
auto_solver/
├── autosolver_agent.py              # 命令行版自动求解器代理
├── dashboard.py                     # Streamlit可视化控制台
├── requirements.txt                 # Python依赖
├── setup_python36_judge_env.bat     # Python3.6评测环境初始化脚本
├── start_dashboard.bat              # Dashboard启动脚本
├── algorithm_components/            # 供LLM参考和组合的算法素材
├── cases/                           # 本地测试案例
├── generated_solvers/               # 最终和中间生成求解器
└── report/                            # 项目报告和图示材料
```

## 环境要求

本项目建议使用Python3.11或更高版本运行代理和Dashboard。如果需要模拟旧版评测环境，可以使用项目中的`setup_python36_judge_env.bat`创建`.venv36`，再通过`--judge-python`指定该解释器。

安装依赖：

```bash
pip install -r requirements.txt
```

依赖主要包括：

- `openai`：调用DeepSeek或其他OpenAI兼容API。
- `ortools`：提供CP-SAT等组合优化能力。
- `streamlit`：运行可视化控制台。
- `plotly`和`pandas`：展示和分析运行记录。

## API密钥配置

默认情况下，程序会读取项目根目录下的`deepseek_api_key.txt`作为API密钥文件。上传GitHub前不建议提交真实密钥，可以改用本地私有文件或环境变量管理。

默认配置如下：

```text
base_url = https://api.deepseek.com
model = deepseek-v4-pro
api_key_file = deepseek_api_key.txt
```

如果使用其他OpenAI兼容服务，可以通过命令行参数覆盖`--base-url`、`--model`和`--api-key-file`。

## 命令行运行

运行全部本地案例并生成3轮候选求解器：

```bash
python autosolver_agent.py --rounds 3
```

只运行指定案例：

```bash
python autosolver_agent.py --case tiny_seed42.txt --rounds 3
```

运行多个案例：

```bash
python autosolver_agent.py --case tiny_seed42.txt --case small_seed100.txt --rounds 5
```

生成并把最佳结果写入`generated_solvers/final_generated.py`：

```bash
python autosolver_agent.py --rounds 5 --apply-best
```

使用指定评测解释器：

```bash
python autosolver_agent.py --rounds 3 --judge-python .venv36/Scripts/python.exe
```

常用参数说明：

| 参数 | 作用 |
|---|---|
| `--case` | 指定单个案例文件，可重复传入 |
| `--case-dir` | 指定案例目录，默认是`cases` |
| `--rounds` | LLM生成轮数 |
| `--out-dir` | 中间生成求解器输出目录 |
| `--runs-dir` | 运行日志和历史记录目录 |
| `--algorithm-dir` | 算法组件目录 |
| `--prompt-mode` | 控制发送给LLM的算法素材数量，可选`selected`、`summary`、`full` |
| `--component-policy` | 组件释放策略，可选`adaptive`、`staged`、`open` |
| `--duel-policy` | 候选对决策略，可选`adaptive`、`stage`、`off` |
| `--timeout` | 单个案例的运行超时时间 |
| `--apply-best` | 将当前最佳生成代码复制到提交路径 |

## Dashboard运行

可以直接使用启动脚本：

```bash
start_dashboard.bat
```

也可以手动启动：

```bash
streamlit run dashboard.py
```

Dashboard会读取`cases`、`agent_runs`和`generated_solvers`目录，用于展示案例、运行状态、日志、候选代码和最佳结果。

## 输入数据格式

案例文件是制表符分隔文本，字段包括`task_id_list`、`courier_id`、`total_score`和`willingness`。

示例：

```text
task_id_list	courier_id	total_score	willingness
T0000	C0006	11.991453	0.610894
T0000	C0001	11.220245	0.432514
T0001	C0006	5.289704	0.539996
```

其中`task_id_list`可以表示单个任务，也可以表示组合任务组；`courier_id`表示可执行该任务组的骑手；`total_score`越低通常越优；`willingness`表示意愿或可靠性信号。

## 评估目标

评估过程优先关注任务覆盖率，其次关注综合成本。代码中使用缺失任务惩罚来避免生成解漏掉大量任务，可近似理解为：

$$
\text{cost}=\text{raw\_cost}+100\times\text{missing\_count}
$$

因此，一个好的求解器需要先尽量覆盖所有任务，再在完整覆盖的候选中降低`total_score`相关成本。

## 输出结果

运行结束后，主要输出包括：

- `generated_solvers/final_generated.py`：可提交或继续测试的最终求解器。
- `agent_runs/<run_id>/best_generated.py`：本次运行中的最佳候选代码。
- `agent_runs/<run_id>/summary.md`：运行摘要。
- `agent_runs/<run_id>/history.jsonl`：每轮生成和评测历史。
- `agent_runs/<run_id>/agent.log`：详细运行日志。

## 算法组件说明

`algorithm_components`目录中的文件不是直接作为基线求解器运行，而是作为LLM可阅读、可改写和可组合的算法材料。

| 文件 | 内容 |
|---|---|
| `01_io_core.py` | 输入解析、数据清洗和核心数据结构 |
| `02_objective.py` | 代理目标函数和结果比较逻辑 |
| `03_initial_builders.py` | 贪心、精确覆盖、单点、配对、打包和CP-SAT初始构造 |
| `04_matching_skeleton.py` | 配对骨架和类似最小费用流的分配辅助逻辑 |
| `05_local_search.py` | 局部替换、骑手重排、LNS和结果打磨 |
| `06_search_orchestration.py` | 退火、搜索调度和历史搜索素材 |

