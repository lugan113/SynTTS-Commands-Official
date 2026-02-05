"""
Keyword Spotting (KWS) Training Pipeline
Target: Bridging the Domain Gap between Synthetic (TTS) and Real-world Audio.
Model: BC-ResNet (Broadcasting Residual Network)
"""

import argparse
import csv
import multiprocessing
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import seaborn as sns
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
    """Broadcasting Residual Block with Depthwise Separable Convolutions."""

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
        # Broadcasting mechanism
        aux = self.aux_pool(x)
        if aux.shape[3] != out.shape[3]:
            aux = F.interpolate(aux, size=(1, out.shape[3]), mode='nearest')
        out = out + self.aux_bn(self.aux_conv(aux))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class BCResNet(nn.Module):
    """Lightweight model for efficient on-device Keyword Spotting."""

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

class AudioDataset(Dataset):
    """Dataset supporting real-time acoustic augmentation."""

    def __init__(self, X_data, y_data, config, is_training=False, noise_pool=None):
        self.X_data = X_data
        self.y_data = y_data
        self.config = config
        self.is_training = is_training
        self.noise_pool = noise_pool
        self.sr = config['sample_rate']
        self.target_len = int(self.sr * config['audio_duration'])

    def __len__(self):
        return len(self.X_data)

    def _normalize_volume(self, audio):
        rms = np.sqrt(np.mean(audio ** 2))
        return audio * (0.15 / (rms + 1e-6))

    def _pitch_shift(self, audio):
        rate = np.random.uniform(1.1, 1.6)
        return scipy.signal.resample(audio, int(len(audio) / rate))

    def _apply_reverb(self, audio):
        delay = int(np.random.uniform(200, 600))
        decay = np.random.uniform(0.2, 0.4)
        reverb = np.zeros(len(audio) + delay)
        reverb[:len(audio)] = audio
        reverb[delay:] += audio * decay
        return reverb[:len(audio)].astype(np.float32)

    def _center_audio(self, audio):
        try:
            non_silent, _ = librosa.effects.trim(audio, top_db=30)
        except:
            non_silent = audio
        if len(non_silent) < 100: non_silent = audio

        if len(non_silent) > self.target_len:
            start = np.random.randint(0, len(non_silent) - self.target_len) if self.is_training else (
                                                                                                                 len(non_silent) - self.target_len) // 2
            return non_silent[start: start + self.target_len]
        else:
            pad_left = (self.target_len - len(non_silent)) // 2
            return np.pad(non_silent, (pad_left, self.target_len - len(non_silent) - pad_left), mode='constant')

    def _add_noise(self, audio):
        if not self.noise_pool: return audio
        noise = random.choice(self.noise_pool)
        if len(noise) <= len(audio):
            noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))[:len(audio)]
        else:
            start = np.random.randint(0, len(noise) - len(audio))
            noise = noise[start: start + len(audio)]
        snr_db = np.random.uniform(10, 20)
        a_rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
        n_rms = np.sqrt(np.mean(noise ** 2)) + 1e-8
        return audio + noise * (a_rms / (10 ** (snr_db / 20)) / n_rms)

    def _audio_to_mel(self, audio):
        # Pre-emphasis
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_fft=512, hop_length=320, n_mels=40)
        log_spec = librosa.power_to_db(spec, ref=np.max).T
        target_h = self.config['input_shape'][0]
        if log_spec.shape[0] > target_h:
            log_spec = log_spec[:target_h, :]
        else:
            log_spec = np.pad(log_spec, ((0, target_h - log_spec.shape[0]), (0, 0)))
        log_spec = (log_spec - np.mean(log_spec)) / (np.std(log_spec) + 1e-6)
        return torch.FloatTensor(log_spec).unsqueeze(0)

    def __getitem__(self, idx):
        audio = self.X_data[idx]
        label = self.y_data[idx]
        audio = self._normalize_volume(audio)

        if self.is_training:
            if self.config['enable_augmentation']:
                if np.random.rand() > 0.4: audio = self._pitch_shift(audio)
                if np.random.rand() > 0.4: audio = self._apply_reverb(audio)
            audio = self._center_audio(audio)
            if self.config['enable_augmentation'] and np.random.rand() > 0.3:
                audio = self._add_noise(audio)
        else:
            audio = self._center_audio(audio)

        return self._audio_to_mel(audio), torch.tensor(label, dtype=torch.long)


# ==========================================
# Loss Functions & Utilities
# ==========================================

class FocalLoss(nn.Module):
    """Focal Loss to address class imbalance."""

    def __init__(self, gamma=2.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        return ((1 - torch.exp(-ce_loss)) ** self.gamma * ce_loss).mean()


def load_single_file(args):
    path, label_idx, target_len, sr = args
    try:
        wav, _ = librosa.load(path, sr=sr)
        if len(wav) > target_len:
            wav = wav[:target_len]
        else:
            wav = np.pad(wav, (0, target_len - len(wav)), mode='constant')
        return wav.astype(np.float32), label_idx
    except:
        return np.zeros(target_len, dtype=np.float32), label_idx


# ==========================================
# Main Training Logic
# ==========================================

def main(args):
    # 1. Configuration
    config = {
        'train_data_dir': args.train_dir,
        'test_data_dir': args.test_dir,
        'noise_data_dir': args.noise_dir,
        'enable_augmentation': not args.no_aug,
        'real_data_oversample_rate': args.oversample,
        'few_shot_count': args.few_shot,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'sample_rate': 16000,
        'audio_duration': 2.0,
        'input_shape': (99, 40),
        'target_labels': ["Hi_Xiaowen", "Negative", "Nihao_Wenwen"],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': args.workers,
        'pin_memory': True,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("results", f"experiment_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    # 2. Mixed Data Loading (Few-shot + Oversampling)
    print(f"🔄 Loading data from {config['train_data_dir']}...")
    train_f, train_y, test_f, test_y = [], [], [], []
    label_to_idx = {l: i for i, l in enumerate(config['target_labels'])}

    for lbl in config['target_labels']:
        p_syn = os.path.join(config['train_data_dir'], lbl)
        p_real = os.path.join(config['test_data_dir'], lbl)

        fs_syn = glob(os.path.join(p_syn, "*.wav")) if os.path.exists(p_syn) else []
        fs_real = shuffle(glob(os.path.join(p_real, "*.wav")), random_state=42) if os.path.exists(p_real) else []

        train_f.extend(fs_syn)
        train_y.extend([label_to_idx[lbl]] * len(fs_syn))

        if len(fs_real) > config['few_shot_count']:
            r_train = fs_real[:config['few_shot_count']]
            r_test = fs_real[config['few_shot_count']:config['few_shot_count'] + 200]
            # Oversampling real samples to balance synthetic data influence
            train_f.extend(r_train * config['real_data_oversample_rate'])
            train_y.extend([label_to_idx[lbl]] * (len(r_train) * config['real_data_oversample_rate']))
            test_f.extend(r_test)
            test_y.extend([label_to_idx[lbl]] * len(r_test))
        else:
            test_f.extend(fs_real)
            test_y.extend([label_to_idx[lbl]] * len(fs_real))

    def parallel_load(files, labels):
        t_len = int(config['sample_rate'] * config['audio_duration'])
        tasks = [(f, l, t_len, config['sample_rate']) for f, l in zip(files, labels)]
        X, y = [None] * len(tasks), [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=12) as ex:
            futures = {ex.submit(load_single_file, t): i for i, t in enumerate(tasks)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading Audio"):
                idx = futures[future]
                X[idx], y[idx] = future.result()
        return np.array(X), np.array(y)

    X_train, y_train = parallel_load(train_f, train_y)
    X_test, y_test = parallel_load(test_f, test_y)

    # Load Noise Pool
    noise_pool = []
    if config['enable_augmentation'] and os.path.exists(config['noise_data_dir']):
        noise_files = glob(os.path.join(config['noise_data_dir'], "*.wav"))[:500]
        for f in tqdm(noise_files, desc="Loading Noise"):
            try:
                n, _ = librosa.load(f, sr=config['sample_rate'])
                if len(n) > config['sample_rate']: noise_pool.append(n)
            except:
                continue

    # 3. DataLoaders
    train_ds = AudioDataset(X_train, y_train, config, is_training=True, noise_pool=noise_pool)
    test_ds = AudioDataset(X_test, y_test, config, is_training=False)

    class_counts = np.bincount(y_train)
    weights = 1. / (class_counts + 1e-6)
    sampler = WeightedRandomSampler(torch.from_numpy(weights[y_train]).double(), len(y_train))

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], sampler=sampler,
                              num_workers=config['num_workers'], pin_memory=config['pin_memory'])
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False,
                             num_workers=config['num_workers'], pin_memory=config['pin_memory'])

    # 4. Training Loop
    model = BCResNet(len(config['target_labels'])).to(config['device'])
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    history = []
    best_acc = 0.0

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(config['device']), labels.to(config['device'])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(config['device'])
                outputs = model(inputs)
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_true.extend(labels.numpy())

        val_acc = accuracy_score(val_true, val_preds)
        print(f"✨ Validation Accuracy: {val_acc:.4f}")
        history.append([epoch, running_loss / len(train_loader), correct / total, val_acc])

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(result_dir, "best_model.pth"))

    # 5. Save Results
    with open(os.path.join(result_dir, "history.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'train_acc', 'val_acc'])
        writer.writerows(history)

    # Final Confusion Matrix
    cm = confusion_matrix(val_true, val_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config['target_labels'], yticklabels=config['target_labels'])
    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
    print(f"✅ Training complete. Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KWS Domain Adaptation Training")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to synthetic training data")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to real test data")
    parser.add_argument("--noise_dir", type=str, default="", help="Path to noise pool")
    parser.add_argument("--few_shot", type=int, default=100, help="Real samples per class in training")
    parser.add_argument("--oversample", type=int, default=100, help="Oversampling rate for real samples")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers (0 for Windows)")
    parser.add_argument("--no_aug", action="store_true", help="Disable data augmentation")

    args = parser.parse_args()
    multiprocessing.freeze_support()
    main(args)