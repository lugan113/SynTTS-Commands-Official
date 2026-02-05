"""
Google Speech Commands (GSC) Domain Adaptation Training
Strategy: Bridging the gap between Synthetic (TTS) and Real-world audio
          using Few-shot Learning and Aggressive Oversampling.
Model: BC-ResNet (Broadcasting Residual Network)
"""

import argparse
import csv
import json
import multiprocessing
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import librosa
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


# ==========================================
# Model Definition: BC-ResNet
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
        out = self.conv1(x)
        out = self.bn1(out)
        aux = self.aux_pool(x)
        if aux.shape[3] != out.shape[3]:
            aux = F.interpolate(aux, size=(1, out.shape[3]), mode='nearest')
        aux = self.aux_conv(aux)
        aux = self.aux_bn(aux)
        out = out + aux
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
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
# Data Processing & Augmentation
# ==========================================

class GSCDataset(Dataset):
    """Dataset class with on-the-fly acoustic augmentation."""

    def __init__(self, X_data, y_data, config, is_training=False, noise_pool=None):
        self.X_data = X_data
        self.y_data = y_data
        self.is_training = is_training
        self.noise_pool = noise_pool
        self.config = config
        self.target_len = int(config.sample_rate * config.audio_duration)

    def __len__(self):
        return len(self.X_data)

    def _augment(self, audio):
        # 1. Pitch Shift (Resampling)
        if random.random() > 0.5:
            rate = random.uniform(1.1, 1.5)
            audio = scipy.signal.resample(audio, int(len(audio) / rate))

        # 2. Simple Reverb
        if random.random() > 0.5:
            delay = int(random.uniform(100, 400))
            decay = random.uniform(0.2, 0.4)
            reverb = np.zeros(len(audio) + delay)
            reverb[:len(audio)] = audio
            reverb[delay:] += audio * decay
            audio = reverb[:len(audio)]

        # 3. Alignment
        if len(audio) > self.target_len:
            start = random.randint(0, len(audio) - self.target_len)
            audio = audio[start:start + self.target_len]
        else:
            pad = self.target_len - len(audio)
            audio = np.pad(audio, (pad // 2, pad - pad // 2))

        # 4. Background Noise Injection
        if self.noise_pool and random.random() < 0.7:
            noise = random.choice(self.noise_pool)
            start = random.randint(0, len(noise) - self.target_len)
            noise_seg = noise[start:start + self.target_len]
            snr_db = random.uniform(5, 15)
            a_rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
            n_rms = np.sqrt(np.mean(noise_seg ** 2)) + 1e-8
            audio = audio + noise_seg * (a_rms / (10 ** (snr_db / 20)) / n_rms)

        return audio.astype(np.float32)

    def _to_mel(self, audio):
        # Volume Normalization
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 1e-6:
            audio = audio * (0.15 / rms)

        # Feature Extraction
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=512, hop_length=320, n_mels=40)
        log_spec = librosa.power_to_db(spec, ref=np.max).T

        # Shape Alignment
        target_h = self.config.input_shape[0]
        if log_spec.shape[0] > target_h:
            log_spec = log_spec[:target_h, :]
        else:
            log_spec = np.pad(log_spec, ((0, target_h - log_spec.shape[0]), (0, 0)))

        # Standardization
        log_spec = (log_spec - np.mean(log_spec)) / (np.std(log_spec) + 1e-6)
        return torch.FloatTensor(log_spec).unsqueeze(0)

    def __getitem__(self, idx):
        audio = self.X_data[idx]
        if self.is_training:
            audio = self._augment(audio)
        else:
            if len(audio) > self.target_len:
                audio = audio[:self.target_len]
            else:
                audio = np.pad(audio, (0, self.target_len - len(audio)))

        return self._to_mel(audio), torch.tensor(self.y_data[idx], dtype=torch.long)


# ==========================================
# Utilities
# ==========================================

def load_audio_parallel(file_list, labels_list, label_map):
    """Parallel audio loading using ThreadPoolExecutor."""

    def _load(f, lbl):
        try:
            wav, _ = librosa.load(f, sr=16000)
            return wav.astype(np.float32), label_map[lbl]
        except Exception:
            return None

    X, y = [], []
    with ThreadPoolExecutor(max_workers=12) as ex:
        tasks = [ex.submit(_load, f, l) for f, l in zip(file_list, labels_list)]
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Loading Audio"):
            res = future.result()
            if res:
                X.append(res[0])
                y.append(res[1])
    return X, y


# ==========================================
# Main Execution
# ==========================================

def main(args):
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"GSC_FewShot_{timestamp}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    label_map = {l: i for i, l in enumerate(args.target_labels)}

    # 1. Scan Files
    train_fs, train_ls = [], []
    test_fs, test_ls = [], []

    print("📂 Scanning directories...")
    for lbl in args.target_labels:
        # Synthetic data
        p_syn = os.path.join(args.train_dir, lbl)
        fs_syn = glob(os.path.join(p_syn, "*.wav")) if os.path.exists(p_syn) else []
        train_fs.extend(fs_syn)
        train_ls.extend([lbl] * len(fs_syn))

        # Real data
        p_real = os.path.join(args.test_dir, lbl)
        if os.path.exists(p_real):
            fs_real = shuffle(glob(os.path.join(p_real, "*.wav")), random_state=42)
            # Few-Shot Split
            r_train = fs_real[:args.few_shot]
            r_test = fs_real[args.few_shot:args.few_shot + 300]

            # Apply Oversampling to real training samples
            train_fs.extend(r_train * args.oversample)
            train_ls.extend([lbl] * (len(r_train) * args.oversample))
            test_fs.extend(r_test)
            test_ls.extend([lbl] * len(r_test))

    # 2. Load Audio
    X_train, y_train = load_audio_parallel(train_fs, train_ls, label_map)
    X_test, y_test = load_audio_parallel(test_fs, test_ls, label_map)

    # Load Noise Pool
    noise_pool = []
    if os.path.exists(args.noise_dir):
        noise_files = glob(os.path.join(args.noise_dir, "*.wav"))
        for f in noise_files[:20]:
            try:
                n, _ = librosa.load(f, sr=16000)
                if len(n) > 16000:
                    noise_pool.append(n)
            except Exception:
                continue

    # 3. DataLoaders
    train_ds = GSCDataset(X_train, y_train, args, is_training=True, noise_pool=noise_pool)
    test_ds = GSCDataset(X_test, y_test, args, is_training=False)

    class_counts = np.bincount(y_train)
    weights = 1. / (class_counts + 1e-6)
    sampler = WeightedRandomSampler(torch.from_numpy(weights[y_train]).double(), len(y_train))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 4. Model Initialization
    model = BCResNet(len(args.target_labels)).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct / total:.3f}")

        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(args.device)
                outputs = model(inputs)
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_true.extend(labels.numpy())

        val_acc = accuracy_score(val_true, val_preds)
        print(f"✨ Epoch {epoch} Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))
            torch.save(model.state_dict(), "latest_gsc_model.pth")

    print(f"✅ Training Complete. Best Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSC Domain Adaptation Training")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to synthetic data")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to real data")
    parser.add_argument("--noise_dir", type=str, default="", help="Path to noise pool")
    parser.add_argument("--few_shot", type=int, default=50, help="Real samples per class in training")
    parser.add_argument("--oversample", type=int, default=50, help="Oversampling rate for real samples")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--audio_duration", float, default=1.0)
    parser.add_argument("--input_shape", type=int, nargs=2, default=[50, 40])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Define labels as a list
    parser.add_argument("--target_labels", type=str, nargs="+", default=[
        "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
        "zero", "one", "three", "dog"
    ])

    args = parser.parse_args()
    multiprocessing.freeze_support()
    main(args)