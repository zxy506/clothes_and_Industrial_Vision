# Industrial Vision

基于条件 VAE 的工业表面缺陷小样本数据增强与智能分类课程项目。已完成环境、Fashion-MNIST 课程实验、NEU-CLS 工业对照实验、本地 Streamlit 软件与最终报告。研究结果保留了 CVAE 增强在本协议下不如常规增强的负结果。

## 启动本地软件

在本项目目录运行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器访问 `http://localhost:8501`。七个页面提供首页、工业缺陷识别、衣物识别、CVAE 样本生成、真实实验结果对比、SQLite 历史查询与 CSV 导出、系统设置。工业识别可选择 NEU-CLS A/B/C 与种子 0/1/2，输入转为 128×128 灰度，输出六类概率；衣物识别可选择 Fashion-MNIST A/B/C25/C50 与种子 0/1/2，输入转为 28×28 灰度，输出十类概率。两页均支持单张或批量 PNG、JPEG、BMP，每次最多处理前 50 张，每张不超过 20 MB。

生成图片及元数据保存在本项目的 `generated/app/`，检测历史保存在 `database/history.db`。模型权重从 `checkpoints/neu/` 加载，界面指标从 `results/summary/` 中的正式实验文件读取。首次加载模型可能稍慢；缺少权重或结果文件时页面会提示。软件推理和生成均在本机运行，不调用外部 API。

可从本项目 `demo_samples/neu_test/` 和 `demo_samples/fashion_test/` 分别选择工业六类、衣物十类的**独立测试集**图片进行演示；具体操作和离线首页截图见 `reports/demo/README.md` 与 `reports/demo/home_with_fashion.png`。衣物模型主要适用于 Fashion-MNIST 风格的黑底、居中、单件灰度衣物；普通照片可能明显失准。工业模型没有“正常”类、未知类拒识或概率校准，不能直接用于生产线安全决策。NEU 原始图像及演示样本不应公开再分发。

软件服务集成检查：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## 环境

在 Windows PowerShell 中，从本目录运行：

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

安装 CUDA 版本 PyTorch 后，再安装其余项目依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

依赖与实际安装状态以 `WORKLOG.md` 为准。`PROJECT_SPEC.md` 是完整用户规格，`DECISIONS.md` 记录固定参数。

## 数据

Fashion-MNIST 使用 torchvision 官方接口下载。划分脚本输出原始训练集索引，测试集保持官方测试集：

```powershell
.\.venv\Scripts\python.exe -m src.datasets.split_builder
```

NEU-CLS 官方页面：https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/ 。该页的 **NEU-CLS** 链接指向 Google Drive 的 `NEU-CLS.rar`。当前已经从此链接取得原始归档并解压到 `data/raw/neu_cls/NEU-CLS/`，来源和 SHA-256 记在 `data/raw/neu_cls/source.json`。重新获取时，可运行 `scripts/download_neu.py`；脚本拒绝覆盖已有归档。不要误用 NEU-DET 或 NEU-CLS-64。验证与划分命令：

```powershell
.\.venv\Scripts\python.exe -m src.datasets.neu_dataset
```

脚本检查六类各 300 张、200×200 灰度图，并输出 180/60/60 划分及 90/类小样本训练索引。原始数据不随项目公开分发；官方页面未明确给出完整再分发许可。

运行 `.\.venv\Scripts\python.exe scripts\preview_datasets.py` 可从训练集重新生成样本网格。后续实验输出只能来自实际运行。

## Fashion 初步训练

```powershell
.\.venv\Scripts\python.exe -m src.training.train_cnn --mode A --epochs 10 --seed 0
.\.venv\Scripts\python.exe -m src.training.train_cvae --epochs 20 --seed 0
.\.venv\Scripts\python.exe -m src.training.train_cnn --mode B --epochs 10 --seed 0
.\.venv\Scripts\python.exe -m src.training.train_cnn --mode C --epochs 10 --seed 0
.\.venv\Scripts\python.exe -m scripts.diagnose_fashion_cvae
.\.venv\Scripts\python.exe scripts\plot_stage2.py
```

这些命令已有对应日志和权重，默认拒绝覆盖。复跑时需使用新的运行标识或在保留原结果后调整输出路径；不要删除原始记录。初步结果在 `results/summary/fashion_stage2_preliminary.json`，训练逐轮记录在 `logs/`。正式实验的测试集未用于模型选择。

## Fashion 正式结果

正式结果及限制见 `reports/fashion_final.md`；机器可读汇总为 `results/summary/fashion_final.json`，逐次混淆矩阵和预测在 `results/raw/fashion_final/`。主要复现命令：

```powershell
.\.venv\Scripts\python.exe -m scripts.run_fashion_formal
.\.venv\Scripts\python.exe -m src.evaluation.evaluate_fashion
.\.venv\Scripts\python.exe scripts\analyze_fashion_final.py
.\.venv\Scripts\python.exe -m scripts.measure_fashion_costs
.\.venv\Scripts\python.exe -m pytest tests -q
```

正式训练前脚本冻结了划分、配置、训练代码及 CVAE 权重的 SHA-256。以上正式输出已存在，脚本会拒绝覆盖测试结果；如需完整重新复现，应在新工作副本中运行。测试集只在全部十二次分类训练后评价。

## NEU-CLS 工业实验

结果及限制见 `reports/neu_final.md`，机器可读汇总为 `results/summary/neu_final.json`。NEU 模型与 Fashion 权重完全独立。主要复现命令：

```powershell
.\.venv\Scripts\python.exe -m src.training.train_neu_cvae
.\.venv\Scripts\python.exe -m scripts.diagnose_neu_cvae
.\.venv\Scripts\python.exe -m scripts.run_neu_formal
.\.venv\Scripts\python.exe -m src.evaluation.evaluate_neu
.\.venv\Scripts\python.exe scripts\analyze_neu_final.py
.\.venv\Scripts\python.exe -m scripts.measure_neu_costs
```

以上结果文件已存在，训练和测试脚本拒绝覆盖；重新跑完整实验请使用新工作副本。NEU 原始数据留在本项目 `data/raw/neu_cls/`，不公开分发。

## 最终报告与验收

- 中文课程报告 PDF：`reports/Industrial_Vision_final_report.pdf`；NeurIPS 2026 官方样式、正文 6 页，参考文献与附录 1 页。
- 可编辑 XeLaTeX 源码及官方模板：`reports/paper_source/`，编译说明见其 `README.md`。
- 最终验收状态、已知边界：`reports/acceptance_checklist.md`。
- 原始预测复算和完整测试集权重重新推理的审计：`reports/final_audit.json`。可重新运行：

```powershell
.\.venv\Scripts\python.exe -m scripts.final_audit
```

审计不会覆盖正式训练权重或测试结果。Fashion 冻结协议中的 CNN 源码版本可在 `reports/frozen_sources/fashion_cnn.py` 找到，其 SHA-256 与测试前协议一致；当前共享 `src/models/cnn.py` 后续追加了 NEU 分类器，因此整文件哈希不同。报告作者已填写智新宇；提交前请审核正文及课程要求。答辩 PPT 由用户自行制作。
