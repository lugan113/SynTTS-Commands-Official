## 🛠️ Data Design Philosophy: "Clean Core + Augmentation"

Our current and future datasets provide **high-fidelity, clean synthesized speech**. We intentionally exclude pre-mixed background noise to maximize flexibility for developers.

*   **Clean Core:** We focus on generating diverse speaker prosody and accents using TTS.
*   **Robustness via Augmentation:** As stated in our paper, we recommend developers augment this dataset with domain-specific noise (e.g., **ESC-50** or **UrbanSound8K**) and Room Impulse Responses (RIR) during training to achieve environmental robustness.

# 🚀 Roadmap: Future Command Sets & Domain Expansion

This document outlines the expansion plan for the **SynTTS-Commands** dataset. Our goal is to cover diverse, high-value edge AI scenarios ranging from smart home automation to safety-critical applications.

We invite the community and domain experts to provide feedback on these command lists.

## 🟢 Phase 1: Multimedia Control (Completed)
**Status:** ✅ Published on Hugging Face  
**Focus:** High-fidelity wake words and media playback controls.

| Category | English Commands | Chinese Commands (中文) |
| :--- | :--- | :--- |
| **Playback** | Play, Pause, Resume, Play from start, Repeat song | 播放, 暂停, 继续播放, 从头播放, 单曲循环 |
| **Navigation** | Previous track, Next track, Last song, Skip song, Jump to first track | 上一首, 下一首, 上一曲, 下一曲, 跳到第一首, 播放上一张专辑 |
| **Volume** | Volume up, Volume down, Mute, Set volume to 50%, Max volume | 增大音量, 减小音量, 静音, 音量调到50%, 音量最大 |
| **Call** | Answer call, Hang up, Decline call | 接听电话, 挂断电话, 拒接来电 |
| **Wake Words** | Hey Siri, OK Google, Hey Google, Alexa, Hi Bixby | 小爱同学, Hello 小智, 小艺小艺, 嗨 三星小贝, 小度小度, 天猫精灵 |

---

## 🟡 Phase 2: Smart Home (Planned)
**Status:** 🚧 In Preparation  
**Challenge:** Diverse acoustic environments (reverb, room size) and far-field recognition.

| Category | English Commands | Chinese Commands (中文) |
| :--- | :--- | :--- |
| **Lighting** | Turn on/off lights, Lights on/off, Dim/Brighten lights, Set lights to 50% | 打开/关闭灯, 开/关灯, 调暗/调亮灯光, 灯光亮度调到50% |
| **Appliances** | Turn on/off TV, AC on/off, Open/Close curtain | 打开/关闭电视, 打开/关闭空调, 开/关空调, 打开/关闭窗帘 |
| **Environment** | Set temp to 26 degrees, Increase/Decrease temp, Increase/Decrease humidity | 温度调到26度, 调高/降低温度, 增加/降低湿度 |
| **Security** | Open the door, Close the door | 打开门, 关闭门, 开门, 关门 |

---

## 🟡 Phase 3: In-Vehicle & Automotive (Planned)
**Status:** 🚧 In Preparation  
**Challenge:** **High Noise Robustness**. Models must perform under engine noise, wind noise, and road friction.

| Category | English Commands | Chinese Commands (中文) |
| :--- | :--- | :--- |
| **Engine/Power** | Start/Stop engine, Turn off engine, Turn on/off car | 启动/关闭引擎, 熄火, 启动/关闭车辆 |
| **Access** | Unlock/Lock car, Open/Close trunk, Open/Close window | 解锁车门, 锁车, 打开/关闭后备箱, 打开/关上车窗 |
| **Climate** | Turn on/off AC, Seat warmer, Defrost windshield | 打开/关闭空调, 打开座椅加热, 除霜 |
| **Lighting** | Fog lights on/off, Hazard lights on/off, High/Low beams, Headlights | 打开/关闭雾灯, 打开/关闭双闪, 打开/关闭远光灯, 打开近光灯/大灯 |
| **Wipers** | Turn on/off wipers, Speed up wipers, Spray windshield | 打开/关闭雨刷, 加快雨刷速度, 喷玻璃水 |
| **Navi & Info** | Navigate home/work, Find gas station, Play driving playlist | 导航回家/公司, 找加油站, 播放驾驶歌单 |

---

## 🔴 Phase 4: Urgent Assistance (Planned)
**Status:** 🚧 Proposal Stage  
**Challenge:** **Ultra-Low Latency & High Recall**. Missing a command here is critical; false negatives must be minimized.

| Category | English Commands | Chinese Commands (中文) |
| :--- | :--- | :--- |
| **General Help** | Help me, I need help, Call for help, Emergency | 救命啊, 我需要帮助, 呼叫救援, 紧急情况 |
| **Specific** | Call 911, Call the police, Call the nurse, Get a doctor | 打110, 打120, 叫护士, 叫医生 |
| **Medical** | I need a doctor, Call an ambulance | 我需要医生, 叫救护车 |

---

## 💡 How to Contribute?
We are actively seeking feedback on:
1.  **Missing Commands:** Are there essential commands we missed?
2.  **Phrasing:** Are these natural phrasings for your region?
3.  **Noise Profiles:** Suggestions for background noise simulation (e.g., specific car models, siren sounds).

Please open an Issue or contact us to suggest changes.
