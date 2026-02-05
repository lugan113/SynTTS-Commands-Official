"""
Google Speech Commands (GSC) Model Evaluation Script
Performs inference on a trained BC-ResNet model and generates
comprehensive metrics (Classification Report & Confusion Matrix).
"""

import argparse
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


# ==========================================
# Model Definition (Must match training)
# ==========================================

class BCResBlock(nn.Module):
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

def preprocess_audio(file_path, target_rms=0.15, sr=16000, duration=1.0):
    """Consistent preprocessing pipeline for inference."""
    wav, _ = librosa.load(file_path, sr=sr)

    # Length alignment
    target_len = int(sr * duration)
    if len(wav) > target_len:
        wav = wav[:target_len]
    else:
        wav = np.pad(wav, (0, target_len - len(wav)))

    # Volume normalization
    rms = np.sqrt(np.mean(wav ** 2))
    if rms > 1e-6:
        wav = wav * (target_rms / rms)

    # Feature extraction (Log-Mel Spectrogram)
    wav = np.append(wav[0], wav[1:] - 0.97 * wav[:-1])  # Pre-emphasis
    spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=512, hop_length=320, n_mels=40)
    log_spec = librosa.power_to_db(spec, ref=np.max).T

    # Standardization
    log_spec = (log_spec - np.mean(log_spec)) / (np.std(log_spec) + 1e-6)
    return torch.FloatTensor(log_spec).unsqueeze(0).unsqueeze(0)


def evaluate(args):
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return

    device = torch.device(args.device)
    print(f"🚀 Evaluating on {device}...")

    # Initialize model
    model = BCResNet(len(args.target_labels)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_preds, all_true = [], []

    print(f"🔍 Scanning test directory: {args.test_dir}")
    for label_idx, label in enumerate(args.target_labels):
        path = os.path.join(args.test_dir, label)
        files = glob(os.path.join(path, "*.wav"))

        # Limit samples for faster evaluation if requested
        if args.limit > 0 and len(files) > args.limit:
            files = files[:args.limit]

        for f in tqdm(files, desc=f"Class: {label}"):
            try:
                input_tensor = preprocess_audio(f).to(device)
                with torch.no_grad():
                    out = model(input_tensor)
                    pred = out.argmax(1).item()

                all_preds.append(pred)
                all_true.append(label_idx)
            except Exception as e:
                print(f"⚠️ Error processing {f}: {e}")
                continue

    # --- Metrics Reporting ---
    print("\n📊 Classification Report:")
    report = classification_report(all_true, all_preds, target_names=args.target_labels)
    print(report)

    # --- Confusion Matrix Visualization ---
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=args.target_labels, yticklabels=args.target_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('GSC Domain Adaptation - Final Evaluation')

    output_plot = "evaluation_confusion_matrix.png"
    plt.savefig(output_plot, dpi=300)
    print(f"✅ Confusion matrix saved to: {output_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSC Model Evaluation")
    parser.add_argument("--model_path", type=str, default="latest_gsc_model.pth", help="Path to trained .pth file")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to real-world test data")
    parser.add_argument("--limit", type=int, default=300, help="Max samples per class (0 for all)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target_labels", type=str, nargs="+", default=[
        "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
        "zero", "one", "three", "dog"
    ])

    args = parser.parse_args()
    evaluate(args)