"""
Chinese Keyword Spotting (KWS) Model Evaluation Script
Target Labels: "Hi_Xiaowen", "Nihao_Wenwen", "Negative"
Model: BC-ResNet (Broadcasting Residual Network)

This script evaluates a trained model on real-world recordings, generating
a classification report and a high-resolution confusion matrix.
"""

import argparse
import os
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm


# ==========================================
# Model Definition (Must match training)
# ==========================================

class BCResBlock(nn.Module):
    """Broadcasting Residual Block for efficient context modeling."""

    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.aux_pool = nn.AdaptiveAvgPool2d((1, None))
        self.aux_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.aux_bn = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.SiLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        aux = self.aux_pool(x)
        if aux.shape[3] != out.shape[3]:
            aux = F.interpolate(aux, size=(1, out.shape[3]), mode='nearest')
        out = out + self.aux_bn(self.aux_conv(aux))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class BCResNet(nn.Module):
    """Lightweight BC-ResNet for Keyword Spotting."""

    def __init__(self, num_classes, scale=1.5, dropout=0.5):
        super().__init__()
        base = int(16 * scale)
        self.conv1 = nn.Conv2d(1, base, kernel_size=(5, 5), stride=(1, 2), padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(base)
        self.relu = nn.SiLU()
        self.layer1 = self._make_layer(base, base, 2, stride=1)
        self.layer2 = self._make_layer(base, base * 2, 2, stride=(2, 1))
        self.layer3 = self._make_layer(base * 2, base * 4, 2, stride=(2, 1))
        self.layer4 = self._make_layer(base * 4, base * 8, 2, stride=(2, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base * 8, num_classes)

    def _make_layer(self, in_planes, out_planes, num_blocks, stride):
        layers = [BCResBlock(in_planes, out_planes, stride)]
        for _ in range(1, num_blocks):
            layers.append(BCResBlock(out_planes, out_planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


# ==========================================
# Inference Logic
# ==========================================

def preprocess_audio(file_path, sr=16000, duration=2.0, target_rms=0.15):
    """
    Consistent preprocessing pipeline for inference.
    Includes: Trimming, Centering, Volume Normalization, and Mel-Spectrogram.
    """
    try:
        wav, _ = librosa.load(file_path, sr=sr)

        # 1. Volume Normalization
        rms = np.sqrt(np.mean(wav ** 2))
        if rms > 1e-6:
            wav = wav * (target_rms / rms)

        # 2. Silence Trimming and Centering
        target_len = int(sr * duration)
        non_silent, _ = librosa.effects.trim(wav, top_db=30)
        if len(non_silent) < 100:
            non_silent = wav

        if len(non_silent) > target_len:
            start = (len(non_silent) - target_len) // 2
            wav = non_silent[start: start + target_len]
        else:
            pad_left = (target_len - len(non_silent)) // 2
            wav = np.pad(non_silent, (pad_left, target_len - len(non_silent) - pad_left), mode='constant')

        # 3. Feature Extraction (Log-Mel Spectrogram)
        wav = np.append(wav[0], wav[1:] - 0.97 * wav[:-1])  # Pre-emphasis
        spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=512, hop_length=320, n_mels=40)
        log_spec = librosa.power_to_db(spec, ref=np.max).T

        # 4. Shape Alignment (99 frames for 2.0s)
        if log_spec.shape[0] > 99:
            log_spec = log_spec[:99, :]
        else:
            log_spec = np.pad(log_spec, ((0, 99 - log_spec.shape[0]), (0, 0)))

        # 5. Standardization
        log_spec = (log_spec - np.mean(log_spec)) / (np.std(log_spec) + 1e-6)
        return torch.FloatTensor(log_spec).unsqueeze(0).unsqueeze(0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def evaluate(args):
    print(f"🔍 Initializing Evaluation...")
    print(f"📂 Model Path: {args.model_path}")
    print(f"📂 Test Directory: {args.test_dir}")

    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found at {args.model_path}")
        return

    device = torch.device(args.device)
    model = BCResNet(len(args.target_labels)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_true = []
    all_pred = []
    misclassified = []

    for label_idx, label_name in enumerate(args.target_labels):
        folder_path = os.path.join(args.test_dir, label_name)
        if not os.path.exists(folder_path):
            print(f"⚠️ Warning: Folder {folder_path} not found. Skipping.")
            continue

        files = glob(os.path.join(folder_path, "*.wav")) + glob(os.path.join(folder_path, "*.flac"))

        if args.limit > 0 and len(files) > args.limit:
            files = random.sample(files, args.limit)

        print(f"\n📊 Evaluating Class: {label_name} ({len(files)} samples)")

        for f in tqdm(files):
            input_tensor = preprocess_audio(f)
            if input_tensor is None:
                continue

            with torch.no_grad():
                output = model(input_tensor.to(device))
                pred_idx = torch.argmax(output, dim=1).item()

            all_true.append(label_idx)
            all_pred.append(pred_idx)

            if pred_idx != label_idx:
                misclassified.append({
                    'path': f,
                    'true': label_name,
                    'pred': args.target_labels[pred_idx]
                })

    # --- Metrics ---
    acc = accuracy_score(all_true, all_pred)
    print("\n" + "=" * 50)
    print(f"🏆 Final Evaluation Results")
    print("=" * 50)
    print(f"Overall Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(all_true, all_pred, target_names=args.target_labels))

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=args.target_labels,
                yticklabels=args.target_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Chinese KWS Confusion Matrix (Acc: {acc:.2%})')

    output_plot = "evaluation_results.png"
    plt.savefig(output_plot, dpi=300)
    print(f"\n✅ Confusion matrix saved to: {output_plot}")

    if misclassified:
        print("\n🧐 Sample Misclassifications (First 5):")
        for item in misclassified[:5]:
            print(f"  - File: {os.path.basename(item['path'])}")
            print(f"    True: {item['true']} | Predicted: {item['pred']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chinese KWS Model Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .pth model file")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to real-world test data")
    parser.add_argument("--limit", type=int, default=1000, help="Max samples per class (0 for all)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target_labels", type=str, nargs="+", default=["Hi_Xiaowen", "Negative", "Nihao_Wenwen"])

    args = parser.parse_args()
    evaluate(args)