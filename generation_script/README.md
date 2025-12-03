# Data Generation Script

This script allows you to generate synthetic keyword spotting datasets using **CosyVoice 2**. It supports multi-GPU inference and batch processing.

## Prerequisites

1. Install **CosyVoice** dependencies following the [official repository](https://github.com/FunAudioLLM/CosyVoice).
2. Prepare a folder containing seed audio files (WAV format).

## Usage

### 1. Generate English Dataset
Use the provided English command list:

```bash
python generate_commands.py \
  --input_dir /path/to/seed_wavs \
  --output_dir ./output_english \
  --commands_file assets/commands_en.txt \
  --model_path iic/CosyVoice2-0.5B
```


###  2. Generate Chinese Dataset
Use the provided Chinese command list:

```bash
python generate_commands.py \
  --input_dir /path/to/seed_wavs \
  --output_dir ./output_chinese \
  --commands_file assets/commands_zh.txt \
  --model_path iic/CosyVoice2-0.5B
```


### 3. Generate Custom Keywords
Create your own my_keywords.txt file (one keyword per line) and run:
```bash
python generate_commands.py --commands_file my_keywords.txt ...


