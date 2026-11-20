# 课程报告源文件

`main.tex` 是中文课程报告，直接使用从 [NeurIPS 2026 官方 Call for Papers](https://neurips.cc/Conferences/2026/CallForPapers) 链接下载的 `neurips_2026.sty`，没有修改样式文件。官方归档保存在 `official_template/Formatting_Instructions_For_NeurIPS_2026.zip`，SHA-256 为 `82473931e3ef710fcd3f4a8cd4119b9de32e56825f90f9e5a6d55f2d01b817d9`。本报告使用 `preprint,nonatbib` 模式，仅作为课程报告，不代表 NeurIPS 投稿或录用。

在本目录运行：

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

需要 XeLaTeX、xeCJK 和 `Microsoft YaHei` 字体。图像从项目 `results/figures/` 读取。编译得到 `main.pdf`，交付副本为 `../Industrial_Vision_final_report.pdf`。当前 PDF 共 7 页，其中正文 6 页，参考文献与附录 1 页；已用 Poppler 渲染逐页检查。

报告中的指标来自 `results/summary/` 和逐次原始记录。`scripts/final_audit.py` 会独立重算并比较；Fashion 冻结时的 CNN 源码精确快照位于 `reports/frozen_sources/fashion_cnn.py`，其 SHA-256 与测试前协议一致。
