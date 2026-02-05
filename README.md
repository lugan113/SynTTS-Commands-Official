# Syntts-Commands-Official：On-Device KWS via Synthetic Speech



<br>

<p align="center">
  <strong>Official Implementation of "SynTTS-Commands: A Public Dataset for On-Device KWS via TTS-Synthesized Multilingual Speech"</strong>
</p>

<p align="center">
  <a href="#-introduction">Introduction</a> •
  <a href="#-dataset-overview">Dataset</a> •
  <a href="#-media-command-categories">Command Categories</a> •
  <a href="#-benchmark-results-and-analysis">Benchmarks</a> •
  <a href="#roadmap">Roadmap</a> 
</p>

---

## 📖 Introduction

**SynTTS-Commands** is a large-scale, multilingual (English & Chinese) synthetic speech command dataset designed for **low-power Keyword Spotting (KWS)** tasks. Generated using state-of-the-art TTS technology (CosyVoice 2), it addresses the data scarcity bottleneck in TinyML and Edge AI.

## 🔗 Resources

| Resource | Description |  |
| :--- | :--- | :--- |
| **💾 Dataset** | **384k+** Audio samples (Wave files) |  
| **🧠 Models** | Pre-trained checkpoints for benchmarks | 

**Due to the double-blind review policy, the link to the full dataset (hosted on Hugging Face) is masked to protect author identity.**

**📂 A sample subset is provided in the data_samples/ folder of this repository for inspection.**


## 🏆 Sim-to-Real Benchmarks

We validated the dataset using **BCResNet** on two standard real-world test sets:
1. **English:** [Google Speech Commands (GSC)](https://arxiv.org/abs/1804.03209) test set.
2. **Chinese:** [MobvoiHotwords](https://www.openslr.org/87) (*Hi Xiaowen*, *Nihao Wenwen*, etc.).

Results show that using **SynTTS-Commands** as a foundation with just **50 real samples** achieves production-ready performance:

| Test Dataset | Strategy | Precision | Recall | Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| **Google Speech Commands** | Zero-shot | 0.89 | 0.88 | 88.0% |
| *(English)* | **50-shot** | **0.93** | **0.93** | **93.0%** |
| | | | | |
| **MobvoiHotwords** | Zero-shot | **0.96** | 0.44 | 67.8% |
| *(Chinese)* | **50-shot** | 0.96 | **0.96** | **95.8%** |

> **Key Insight:** On the real-world **MobvoiHotwords** dataset, 50-shot adaptation dramatically improved recall (44% → 96%) while maintaining exceptional precision (Low False Alarm).



## 📊 Dataset Overview

### Statistics

The **SynTTS-Commands-Media-Dataset** contains a total of **384,621 speech samples**, covering **48 distinct multimedia control commands**. It is divided into four subsets with the following distribution:

| Subset | Speakers | Commands | Samples | Duration (hrs) | Size (GB) |
|------|----------|--------|----------|------------|----------|
| Free-ST-Chinese | 855 | 25 | 21,214 | 6.82 | 2.19 |
| Free-ST-English | 855 | 23 | 19,228 | 4.88 | 1.57 |
| VoxCeleb1&2-Chinese | 7,245 | 25 | 180,331 | 58.03 | 18.6 |
| VoxCeleb1&2-English | 7,245 | 23 | 163,848 | 41.6 | 13.4 |
| **Total** | **8,100** | **48** | **384,621** | **111.33** | **35.76** |

### Dataset Highlights

- **Massive Scale**: Totaling **111.33 hours** and **35.76 GB** of synthetic speech data, making it one of the largest synthetic speech command datasets for academic research.
- **Extensive Speaker Diversity**: Covers **8,100 unique speakers**, spanning various accent groups, age ranges, and recording conditions.
- **Multi-Dimensional Research Support**: The four-subset structure enables research into cross-lingual speaker adaptation, speaker diversity effects, and acoustic robustness in different recording environments.
- **Application-Oriented**: Specifically focused on multimedia playback control scenarios, providing high-quality training data for real-world deployment.

## 🎯 Media Command Categories

### English Media Control Commands (23 Classes)

Playback Control: "Play", "Pause", "Resume", "Play from start", "Repeat song"

Navigation: "Previous track", "Next track", "Last song", "Skip song", "Jump to first track"

Volume Control: "Volume up", "Volume down", "Mute", "Set volume to 50%", "Max volume"

Communication: "Answer call", "Hang up", "Decline call"

Wake Words: "Hey Siri", "OK Google", "Hey Google", "Alexa", "Hi Bixby"

### Chinese Media Control Commands (25 Classes)

Playback Control: "播放", "暂停", "继续播放", "从头播放", "单曲循环"

Navigation: "上一首", "下一首", "上一曲", "下一曲", "跳到第一首", "播放上一张专辑"

Volume Control: "增大音量", "减小音量", "静音", "音量调到50%", "音量最大"

Communication: "接听电话", "挂断电话", "拒接来电"

Wake Words: "小爱同学", "Hello 小智", "小艺小艺", "嗨 三星小贝", "小度小度", "天猫精灵"


## 📈 Benchmark Results and Analysis

We present a comprehensive benchmark of **six representative acoustic models** on the SynTTS-Commands-Media Dataset across both English (EN) and Chinese (ZH) subsets. All models are evaluated in terms of **classification accuracy**, **cross-entropy loss**, and **parameter count**, providing insights into the trade-offs between performance and model complexity in multilingual voice command recognition.

### Performance Summary

| Model | EN Loss | EN Accuracy | EN Params | ZH Loss | ZH Accuracy | ZH Params |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MicroCNN** | 0.2304 | 93.22% | 4,189 | 0.5579 | 80.14% | 4,255 |
| **DS-CNN** | 0.0166 | 99.46% | 30,103 | 0.0677 | 97.18% | 30,361 |
| **TC-ResNet** | 0.0347 | 98.87% | 68,431 | 0.0884 | 96.56% | 68,561 |
| **CRNN** | **0.0163** | **99.50%** | 1.08M | 0.0636 | **97.42%** | 1.08M |
| **MobileNet-V1** | 0.0167 | **99.50%** | 2.65M | **0.0552** | 97.92% | 2.65M |
| **EfficientNet** | 0.0182 | 99.41% | 4.72M | 0.0701 | 97.93% | 4.72M |




## <span id="roadmap"></span>🗺️ Roadmap & Future Expansion

We are expanding SynTTS-Commands beyond multimedia to support broader Edge AI applications. 

👉 **[Click here to view our detailed Future Work Plan & Command List](Future_Work_Plan.md)**

Our upcoming domains include:
*   🏠 **Smart Home:** Far-field commands for lighting and appliances.
*   🚗 **In-Vehicle:** Robust commands optimized for high-noise driving environments.
*   🚑 **Urgent Assistance:** Safety-critical keywords (e.g., "Call 911", "Help me") focusing on high recall.

We invite the community to review our [Command Roadmap](Future_Work_Plan.md) and suggest additional keywords!







