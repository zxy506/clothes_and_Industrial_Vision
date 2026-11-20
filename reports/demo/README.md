# 本地软件演示材料

## 启动

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器打开 `http://localhost:8501`。若 8501 端口已被占用，可使用 `--server.port 8502`，并访问对应端口。

## 识别演示

进入“工业缺陷识别”，选 A · 真实数据、模型种子 0。上传项目 `demo_samples/neu_test/` 内的六张 BMP；它们分别来自 NEU-CLS **独立测试集**，并非训练集。点击“开始识别”。2026-11-19 本机界面实测：Cr、In、Pa、PS、RS、Sc 六张均预测为对应类别，保存 6 条历史记录。单张 `Cr_61.bmp` 也已独立上传并成功识别，另存 1 条历史记录。页面可查看六类概率、下载本次 CSV；“历史检测记录”页可查询、筛选和导出 CSV。

| 文件 | 真实类别 | A/seed 0 页面预测 |
|---|---|---|
| `Cr_61.bmp` | 龟裂 Cr | 龟裂 Cr |
| `In_198.bmp` | 夹杂 In | 夹杂 In |
| `Pa_178.bmp` | 斑块 Pa | 斑块 Pa |
| `PS_281.bmp` | 麻点 PS | 麻点 PS |
| `RS_249.bmp` | 轧入氧化皮 RS | 轧入氧化皮 RS |
| `Sc_261.bmp` | 划痕 Sc | 划痕 Sc |

这六张只用于演示，不能据此估计整体准确率。正式测试统计使用每种子完整 360 张 NEU 测试图，见 `results/summary/neu_final.json`。

## 生成演示

进入“CVAE 样本生成”，选 Cr、数量 6、随机种子 0，点击“生成样本”。训练好的 NEU-CVAE 输出 128×128 灰度合成图，图片和 `metadata.json` 保存到项目 `generated/app/`。本次界面实际生成的样本位于 `generated/app/20261119_194315_ddd033f2/`。合成图只表示模型学到的类别条件纹理，不是上传图片的修复结果，也不保证保留真实龟裂细节；软件可下载 ZIP。

## 衣物识别演示

进入“衣物识别”，可选 A/B/C25/C50 和随机种子 0/1/2。上传项目 `demo_samples/fashion_test/` 中的 PNG；文件名开头五位是 Fashion-MNIST **官方测试集**索引，每类首张各一张。衣物页展示十类中文与英文名称、十类预测概率，保存到同一项目 SQLite，并支持本次 CSV 导出；历史页用“任务”选择衣物记录。服务层已经用真实 B/seed 0 权重确认这些样本的预测与正式测试原始预测一致。衣物页浏览器自动化上传尚未完成验收，演示时请手动上传确认。普通生活照片与 Fashion-MNIST 黑底单件衣物分布不同，预测可能不准。

## 离线材料

- `reports/demo/home_with_fashion.png`：2026-11-20 从实际运行中的 Streamlit 首页截取，展示七页导航、两套分类权重状态和真实实验摘要；`home.png` 保留拓展前版本。
- `demo_samples/neu_test/`：六类独立测试图片及来源说明。原始 NEU 图像不应随公开项目再分发；官方页面未明确给出完整再分发许可。
- `demo_samples/fashion_test/`：十类 Fashion-MNIST 官方测试图片，每类首张用于衣物页演示。
- `results/figures/neu_cvae_diagnostic.png`：真实训练图、生成图与像素最近邻对照，可在软件不可用时解释生成质量。
- `results/figures/neu_final_abc.png`：正式 NEU A/B/C 结果图。
