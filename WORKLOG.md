# 执行日志

## 2026-11-19：第一阶段环境与数据准备完成

- 已检查工作区：原有文件仅 `深度学习课程结课大作业任务书.pptx`，未修改。
- 已读取课程任务书中课题一的关键要求；Fashion-MNIST 500/类训练、500/类验证，官方测试集保留；A/B/C 与 C25、三个训练种子均为必做。
- 本机检测：Windows 11，24 个逻辑处理器，物理内存约 15.7 GiB（当时可用约 4.1 GiB），D 盘可用约 100.4 GiB；`python` 为 Anaconda Python 3.13.5，pip 25.1，Git 2.49.0；`nvidia-smi` 检出 RTX 4060 Laptop GPU 8 GB、驱动 616.56。CIM 查询遭环境拒绝，改用 `platform`、`psutil` 与 `shutil` 获取系统信息。
- 全局用户环境的 PyTorch 2.8.0+cu126 可以识别 CUDA 和 RTX 4060；这仅是诊断，不代表项目虚拟环境已验证。
- 已建立 `Industrial_Vision/` 项目目录，复制完整用户规格至 `PROJECT_SPEC.md`，创建独立虚拟环境。
- 项目虚拟环境安装 PyTorch 2.8.0+cu126、torchvision 0.23.0+cu126 和 `requirements.txt` 的依赖。`scripts/check_env.py` 在 CUDA 上执行 `2×3=6` 张量运算成功，详情见 `logs/environment.json`；精确包版本见 `requirements-lock.txt`。
- 使用 torchvision 官方接口下载 Fashion-MNIST；已生成 `data/splits/fashion.json`，训练和验证各 5000 张，每类各 500，索引交叉数 0，官方 10000 张测试集独立保留。划分清单 SHA-256：`a90236d632fe75b69e74db0265eef6ea569c57c402966563c89cd8d131e54cc4`。
- 划分单元测试：`python -m pytest tests/test_split_builder.py -q`，2 项通过。
- 东北大学官方页面提供 NEU-CLS 的 Google Drive 和百度网盘入口。通过官方 Google Drive 下载确认页取得 `NEU-CLS.rar`（50,733,617 字节；SHA-256 `a85bcc6bfa1a40e76f28079e135694aaf5f6a4ee01203f7eaacff0ebafaa96f8`），归档及解压图像均在本项目的 `data/raw/neu_cls/`。来源记录在 `source.json`；官方页未明确列出完整再分发许可。
- 已校验 NEU 六类各 300 张 200×200 灰度图，生成 `data/splits/neu.json`：训练 1080、验证 360、测试 360，小样本训练 540（90/类），集合零交叉。清单 SHA-256：`802fdaf4edc67d8a42c7b3e6ae4d98d480ed0a9cd2407f9fc8868f152c2e2a5f`。修复了 JSON 元组/列表比较导致的复跑误报，复跑现成功。
- 已生成并目视检查 `results/figures/fashion_train_samples.png` 与 `neu_train_samples.png`，样本均来自训练划分。
- 项目虚拟环境执行 `python -m pytest tests -q`：2 项通过；Fashion 与 NEU 划分脚本复跑均成功。
- 尚未完成模型训练或实验；本日志不将其列为成果。

### 主要执行命令

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\check_env.py
.\.venv\Scripts\python.exe -m src.datasets.split_builder
.\.venv\Scripts\python.exe scripts\download_neu.py
tar -xf data\raw\neu_cls\NEU-CLS.rar -C data\raw\neu_cls
.\.venv\Scripts\python.exe -m src.datasets.neu_dataset
.\.venv\Scripts\python.exe -m pytest tests -q
```

### 下一阶段

实现 CNN、CVAE 和训练日志管线，先做 Fashion-MNIST 基线训练；NEU 使用独立模型。保持固定划分与测试集隔离。

## 2026-11-19：第二阶段课程核心算法初步验证完成

- 实现 Fashion 自建 CNN、条件卷积 VAE、数据加载和训练脚本。标签同时进入 CVAE 编码器与解码器；输出为 28×28 灰度图。
- 模型形状、损失反向梯度检查通过；`.venv` 运行 `python -m pytest tests -q`，3 项通过。
- CNN seed 0 A/B/C 各训练 10 轮、每轮 40 次更新。最佳**验证**准确率：A 0.8194（第 9 轮）、B 0.8048（第 8 轮）、C 0.7878（第 9 轮）。逐轮日志和最佳验证权重均已保存；未评估测试集。
- Fashion CVAE seed 0 训练 20 轮；最后一轮平均总损失 248.089、重建 234.340、KL 13.749（像素与潜在维度先求和、再按批平均）。生成 5000 张，500/类；生成池记录了种子、权重 SHA-256 和耗时。
- 生成样本与训练集最近邻对照、训练曲线和 A/B/C 验证曲线已保存。观察到若干类别的生成样本模糊和混淆，详见 `reports/fashion_preliminary.md`。
- 运行诊断脚本时首次使用 `python scripts/diagnose_fashion_cvae.py` 遇到模块路径错误，改用 `python -m scripts.diagnose_fashion_cvae` 后成功。初步实验后修正 B/C 随机采样流，正式实验将使用新实现；既有日志不覆盖。
- 当前结果仅一颗随机种子和 10 轮预实验，不是正式测试结论。下一阶段进行 0/1/2 三种子、C25、鲁棒性和测试集评价。

### 主要执行命令

```powershell
.\.venv\Scripts\python.exe -m src.training.train_cnn --mode A --epochs 10 --seed 0
.\.venv\Scripts\python.exe -m src.training.train_cvae --epochs 20 --seed 0
.\.venv\Scripts\python.exe -m src.training.train_cnn --mode B --epochs 10 --seed 0
.\.venv\Scripts\python.exe -m src.training.train_cnn --mode C --epochs 10 --seed 0
.\.venv\Scripts\python.exe -m scripts.diagnose_fashion_cvae
.\.venv\Scripts\python.exe scripts\plot_stage2.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

## 2026-11-19：第三阶段 Fashion 正式实验完成

- 在首次读取官方测试集前，将 Fashion 划分、配置、CNN 与训练代码、CVAE 权重的 SHA-256 及 σ=0.15 写入 `results/raw/fashion_formal_protocol.json`。
- 完成 A/B/C25/C50 × seed 0/1/2 共十二次分类训练，每次 30 轮、每轮 40 次更新；验证集选择最佳轮次。逐轮日志、最佳权重、生成池及来源哈希均已保存。十二次全部结束后才执行独立测试。
- 三种子干净测试 Accuracy 均值±样本标准差：A 0.8365±0.0110，B 0.8442±0.0026，C25 0.8301±0.0049，C50 0.8155±0.0024；Macro-F1：0.8350±0.0098、0.8454±0.0022、0.8290±0.0063、0.8147±0.0006。其余逐次、逐类与加噪数值见 `reports/fashion_final.md` 和 `results/summary/fashion_final.json`。
- 固定高斯噪声 σ=0.15 的测试完成，全部模式有明显性能下降；原始预测和混淆矩阵保存于 `results/raw/fashion_final/`。已生成 A/B/C、生成比例消融、逐类召回和混淆矩阵图。
- 已生成每类 8 张的 CVAE 网格、训练最近邻对照及 3 个真实测试失败案例。案例索引和预测可追溯到保存的原始结果。
- 成本测量见 `results/summary/fashion_costs.json`，包含 CVAE/CNN 训练、批量与单张生成、单张推理和 GPU 峰值张量分配。计时范围不含进程启动和数据加载。
- 运行 `.venv` 中 `python -m pytest tests -q`，4 项通过；额外审计确认十二次运行均为 30×40 更新、完整日志及预测文件，训练/验证索引无交叉。
- 正式结果显示 B 组在干净集平均优于 A，而 C25 与 C50 低于 A；保留 CVAE 负结果，不重新调整本轮协议。下一阶段为 NEU-CLS 独立工业模型与 A/B/C 验证。

### 主要执行命令

```powershell
.\.venv\Scripts\python.exe -m scripts.run_fashion_formal
.\.venv\Scripts\python.exe -m src.evaluation.evaluate_fashion
.\.venv\Scripts\python.exe scripts\analyze_fashion_final.py
.\.venv\Scripts\python.exe -m scripts.measure_fashion_costs
.\.venv\Scripts\python.exe scripts\plot_fashion_generated_grid.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

## 2026-11-19：第四阶段 NEU 工业应用实验完成

- 使用官方 NEU-CLS 的固定小样本训练集 90 张/类（共 540 张），验证集 60 张/类，测试集 60 张/类。独立实现并训练六分类 CNN 与 128×128 条件卷积 VAE，不使用 Fashion 模型权重。
- NEU CVAE seed 0 训练 40 轮；最后一轮每图像总损失 9950.261、重建 9911.550、KL 38.710，峰值 GPU 张量分配约 109 MB。生成样本常缺少细裂纹与麻点等高频纹理，训练集最近邻对照已保存。
- 常规增强初版预览出现明显黑色旋转边角。正式训练前将旋转范围从 ±10° 收至 ±5°，中灰填充值 0.5，保留水平翻转概率 0.5；初版与修订版预览均保留，后者经目视检查。
- 在首次测试前冻结 NEU 配置、划分、模型与训练代码、CVAE 权重的 SHA-256，见 `results/raw/neu_formal_protocol.json`。完成 A/B/C × seed 0/1/2 共九次训练，每次 40 轮、每轮 17 次更新。全部结束后才评价独立 360 张测试集。
- 三种子测试 Accuracy 均值±样本标准差：A 0.9731±0.0085，B 0.9769±0.0098，C 0.9343±0.0098；Macro-F1：0.9732±0.0084、0.9770±0.0096、0.9343±0.0095。逐次、逐类和混淆矩阵见 `reports/neu_final.md` 与 `results/summary/neu_final.json`。
- 已保存六类生成网格、最近邻、逐类召回与混淆矩阵图，及三个 A 正确而 C 错误的真实测试案例。保留 C 表现下降的负结果。
- 成本测量见 `results/summary/neu_costs.json`。审计确认九次训练与测试输出完整；训练、验证、测试路径无交叉，跨划分原始文件 SHA-256 无完全重复。
- `.venv` 内运行 `python -m pytest tests -q`，5 项通过。下一阶段开发本地 Streamlit 页面，接入实际 NEU 权重和实验结果文件。

### 主要执行命令

```powershell
.\.venv\Scripts\python.exe -m src.training.train_neu_cvae
.\.venv\Scripts\python.exe -m scripts.diagnose_neu_cvae
.\.venv\Scripts\python.exe -m scripts.preview_neu_augmentation
.\.venv\Scripts\python.exe -m scripts.run_neu_formal
.\.venv\Scripts\python.exe -m src.evaluation.evaluate_neu
.\.venv\Scripts\python.exe scripts\analyze_neu_final.py
.\.venv\Scripts\python.exe -m scripts.measure_neu_costs
.\.venv\Scripts\python.exe -m pytest tests -q
```

## 2026-11-19：第五阶段本地软件开发完成

- 新增 `app.py` 与 `ui/pages/` 六个 Streamlit 页面：首页、工业缺陷识别、CVAE 样本生成、实验结果对比、历史检测记录和系统设置。分类与生成均调用第四阶段保存的真实 NEU 权重；实验图表读取第三、四阶段的正式 JSON 结果。
- 分类服务支持 A/B/C × seed 0/1/2、128×128 灰度预处理、六类概率和权重哈希版本。识别页支持单张及批量上传、预览、CSV 导出，单次处理前 50 张图片；损坏图片、超大文件和缺失模型均有错误提示。仅作图像级分类。
- 生成服务使用 NEU-CVAE seed 0 权重，支持类别、数量和随机种子。生成的 PNG 与元数据保存在项目 `generated/app/`，页面可下载 ZIP。
- 历史服务使用项目内 SQLite `database/history.db`，记录检测时间、文件名、类别、概率、模型及版本；可筛选查询并导出 CSV。修复 Windows 下数据库连接未及时关闭导致文件锁定的问题。
- 浏览器实测首页、识别页控件、实验图表、历史页、设置页，并从生成页真实生成六张 Cr 图片；样本与元数据落在 `generated/app/20261119_194315_ddd033f2/`。浏览器文件选择器未完成真实图片上传操作，批量推理和历史保存由真实权重集成测试覆盖。
- `.venv` 运行 `python -m pytest tests -q`：8 项通过，包含两张真实 NEU 图片的批量推理、六类概率和、可重复生成、SQLite 往返与 CSV、损坏上传和缺失权重。下一阶段进行最终集成验收、报告与演示材料。

### 主要执行命令

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

## 2026-11-19：第六阶段最终交付完成

- 新增只读复核脚本 `scripts/final_audit.py`，核验 Fashion/NEU 划分、21 组权重哈希、训练日志、原始预测的 Accuracy/Macro-F1/Recall/混淆矩阵和三种子均值及样本标准差。使用 CUDA 从权重重新推理完整 Fashion B/seed 0 干净及加噪测试集、NEU B/seed 0 测试集，预测与保存文件逐项完全一致。审计输出为 `reports/final_audit.json`，未覆盖正式实验结果。
- 定位 Fashion 冻结协议中 CNN 源码哈希与当前共享文件的差异：后续追加了 NEU 分类器。恢复并保存 Fashion-only 原始源码到 `reports/frozen_sources/fashion_cnn.py`；SHA-256 与测试前协议精确匹配。其余 Fashion 冻结项与 NEU 冻结项均匹配。
- 从 NeurIPS 2026 官方页面取得模板归档，保存在项目 `reports/paper_source/official_template/`；归档 SHA-256 为 `82473931e3ef710fcd3f4a8cd4119b9de32e56825f90f9e5a6d55f2d01b817d9`。使用未经修改的官方 `.sty` 和 XeLaTeX 编译中文报告，正文 6 页、参考文献及附录 1 页；逐页渲染目视检查。交付 PDF 为 `reports/Industrial_Vision_final_report.pdf`。
- 修复首页和设置页部分粗体标签显示为字面 `**` 的问题，浏览器确认修复后标签正常加粗。通过真实界面上传六张独立测试图并批量识别，六类预测均对应真实标签，保存六条历史；另单独上传一张测试图识别成功，再保存一条。历史页可查询这些记录并显示 CSV 导出控件。生成页实际生成六张 Cr 样本已在第五阶段验证。
- 生成项目内演示目录 `demo_samples/neu_test/`，每类复制一张固定独立测试图，原始数据未移动；用实际运行中的 Streamlit 首页制作离线截图 `reports/demo/home.png`，并编写演示说明、报告编译说明和最终验收清单。
- 最终 `.venv` 执行 `python -m pytest tests -q` 全部 8 项通过。未创建或修改答辩 PPT；NEU 原图和演示图不做公开分发。

### 主要执行命令

```powershell
.\.venv\Scripts\python.exe -m scripts.final_audit
.\.venv\Scripts\python.exe -m pytest tests -q
cd reports\paper_source
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

## 2026-11-20：作者信息与衣物识别拓展

- 按用户提供的姓名，将课程报告作者从通用占位改为“智新宇”，并在报告软件章节加入 Fashion 衣物识别说明。XeLaTeX 重新编译后仍为 7 页（正文 6 页），交付 PDF 已更新。
- 新增独立 Fashion-MNIST 十分类推理服务与“衣物识别”页面：正式 30 轮 A/B/C25/C50 × seed 0/1/2 共 12 个权重，28×28 灰度预处理，十类中英文类别与概率、批量上传、本次 CSV 导出。衣物记录沿用项目 SQLite，通过模型类型与工业记录分开查询。
- 从本项目内已有的官方 Fashion-MNIST 测试集按固定“每类首张”规则准备 10 张演示 PNG 于 `demo_samples/fashion_test/`，未使用训练图。真实 B/seed 0 模型对演示图的推理与已保存的正式测试预测一致。
- 首页、系统设置及历史页更新为两个分类任务；七页导航、衣物页控件和 Fashion 权重 12/12 状态在本地浏览器中显示。浏览器文件选择器在上传 Fashion 样本时两次超时，衣物页面端到端上传尚未验收；核心推理、概率与 SQLite 历史由集成测试验证。`.venv` 中 `python -m pytest tests -q` 为 9 项通过。
- 更新 README、演示说明、最终验收清单与技术决策。所有新文件均在项目文件夹内。

## 14 天执行计划（每天约 3–5 小时）

| 天 | 可验收产物 |
|---|---|
| 1 | 环境与 CUDA 验证、项目骨架、配置与日志 |
| 2 | Fashion 下载、固定划分、NEU 官方数据核验及划分 |
| 3 | CNN、Fashion 基线训练与验证曲线 |
| 4 | Fashion CVAE 训练、类别生成网格与损失曲线 |
| 5 | 传统增强及 A/B/C 试运行、固定正式配置 |
| 6 | A/B/C 和 C25，种子 0/1/2 的正式训练 |
| 7 | 鲁棒性、最近邻、失败案例、成本汇总 |
| 8 | NEU 数据与 CNN 基线 |
| 9 | NEU CVAE 训练与质量检查 |
| 10 | NEU A/B/C 与分类分析 |
| 11 | Streamlit 首页、识别与生成 |
| 12 | 实验对比、历史记录、异常处理 |
| 13 | 集成检查、报告源文件初稿 |
| 14 | 关键复现、文档与交付验收 |

优先级：Fashion 必做实验 > NEU 核心实验 > 软件主要功能 > 界面与拓展。实际进度与未完成项持续记录，不预填实验结果。
