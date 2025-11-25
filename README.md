# Syntts-Commands-Official：On-Device KWS via Synthetic Speech

<!-- Badges -->
<div align="center">
  
  [![arXiv](https://img.shields.io/badge/arXiv-2511.07821-b31b1b.svg)](https://arxiv.org/abs/2511.07821)
  [![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-ffd21e)](https://huggingface.co/datasets/lugan/SynTTS-Commands-Media-Dataset)
  [![Benchmarks](https://img.shields.io/badge/🤗%20Hugging%20Face-Benchmarks-ffd21e)](https://huggingface.co/datasets/lugan/SynTTS-Commands-Media-Benchmarks)
  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

<br>

<p align="center">
  <strong>Official Implementation of "SynTTS-Commands: A Public Dataset for On-Device KWS via TTS-Synthesized Multilingual Speech"</strong>
</p>

<p align="center">
  <a href="#-introduction">Introduction</a> •
  <a href="#-resources">Resources</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-benchmark-results">Benchmarks</a> •
  <a href="#-citation">Citation</a>
</p>

---

## 📖 Introduction

**SynTTS-Commands** is a large-scale, multilingual (English & Chinese) synthetic speech command dataset designed for **low-power Keyword Spotting (KWS)** tasks. Generated using state-of-the-art TTS technology (CosyVoice 2), it addresses the data scarcity bottleneck in TinyML and Edge AI.

## 🔗 Resources

| Resource | Description | Link |
| :--- | :--- | :--- |
| **📄 Paper** | Full technical report and analysis | [arXiv:2511.07821](https://arxiv.org/abs/2511.07821) |
| **💾 Dataset** | **384k+** Audio samples (Wave files) | [🤗 HF Dataset](https://huggingface.co/datasets/lugan/SynTTS-Commands-Media-Dataset) |
| **🧠 Models** | Pre-trained checkpoints for benchmarks | [🤗 HF Models](https://huggingface.co/datasets/lugan/SynTTS-Commands-Media-Benchmarks) |


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

### 🔍 Key Findings

Our results demonstrate that the **SynTTS-Commands-Media** dataset supports high-accuracy command recognition in both languages. Notably, the top-performing models achieve over **99.4% accuracy on English** and nearly **98% on Chinese**, confirming the dataset’s quality and suitability for real-world deployment.

- **Top Performers**: Among all models, **CRNN** attains the best English accuracy (**99.50%**) and the lowest loss (0.0163). **MobileNet-V1** yields the lowest loss on Chinese (0.0552) and competitive English performance (matching CRNN’s 99.50% accuracy). Interestingly, **EfficientNet** shows slightly higher Chinese accuracy (97.93%) than MobileNet-V1, suggesting better calibration or robustness despite a higher loss.

- **Accuracy-Complexity Trade-off**: Lightweight models exhibit a clear trade-off. **MicroCNN**, with only ~4.2K parameters, achieves 93.22% accuracy on English but drops to 80.14% on Chinese, highlighting the increased difficulty of modeling tonal and phonetic richness in Mandarin with ultra-compact architectures. DS-CNN and TC-ResNet, with under 70K parameters, already recover strong performance (>96.5% in both languages), underscoring their efficiency for resource-constrained applications.

Overall, the benchmark establishes strong baselines across a wide spectrum of model scales—from ultra-light MicroCNN to modern EfficientNet—demonstrating that moderate-complexity models can deliver near-SOTA performance suitable for edge deployment.

## 📜 Citation

If you use these **pre-trained models** or the **SynTTS-Commands dataset** in your research, please cite our paper:

**[SynTTS-Commands: A Public Dataset for On-Device KWS via TTS-Synthesized Multilingual Speech](https://arxiv.org/abs/2511.07821)**

```bibtex
@misc{gan2025synttscommands,
      title={SynTTS-Commands: A Public Dataset for On-Device KWS via TTS-Synthesized Multilingual Speech}, 
      author={Lu Gan and Xi Li},
      year={2025},
      eprint={2511.07821},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2511.07821}, 
      doi={10.48550/arXiv.2511.07821}
}


