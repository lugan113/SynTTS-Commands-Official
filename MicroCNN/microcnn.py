import tensorflow as tf
import numpy as np
import os
import json
import pandas as pd
from tensorflow.keras import layers, models
import matplotlib

matplotlib.use('Agg')  # Optimize for headless servers
import matplotlib.pyplot as plt
import librosa
import re
from pathlib import Path
from collections import Counter
import argparse
import sys

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class MicroCNNBuilder:
    """MicroCNN Model Builder - Ultra-lightweight CNN for KWS."""

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_micro_cnn(self, model_size='small'):
        """
        Builds MicroCNN model.
        Args:
            model_size: 'nano', 'micro', 'small'
        """
        inputs = tf.keras.Input(shape=self.input_shape)

        # Model Configurations (Filters, Kernel Sizes, Units)
        configs = {
            'nano': {
                'filters': [4, 8, 12, 16],
                'kernel_sizes': [(3, 3), (3, 3), (3, 3), (3, 3)],
                'units': 16
            },
            'micro': {
                'filters': [8, 16, 24, 32],
                'kernel_sizes': [(3, 3), (3, 3), (3, 3), (3, 3)],
                'units': 32
            },
            'small': {
                'filters': [16, 32, 48, 64],
                'kernel_sizes': [(5, 5), (3, 3), (3, 3), (3, 3)],
                'units': 64
            }
        }

        if model_size not in configs:
            raise ValueError(f"Unknown model size: {model_size}")

        cfg = configs[model_size]
        x = inputs

        # Stacked Convolutional Blocks
        for i, (filters, kernel_size) in enumerate(zip(cfg['filters'], cfg['kernel_sizes'])):
            # Depthwise Separable Conv to reduce parameters
            x = layers.DepthwiseConv2D(kernel_size, padding='same', name=f'dw_conv{i + 1}')(x)
            x = layers.BatchNormalization(name=f'bn_dw{i + 1}')(x)
            x = layers.ReLU(name=f'relu_dw{i + 1}')(x)

            # Pointwise Conv
            x = layers.Conv2D(filters, (1, 1), padding='same', name=f'pw_conv{i + 1}')(x)
            x = layers.BatchNormalization(name=f'bn_pw{i + 1}')(x)
            x = layers.ReLU(name=f'relu_pw{i + 1}')(x)

            # Pooling (skip for last layer to keep some spatial dim before GAP)
            if i < len(cfg['filters']) - 1:
                x = layers.MaxPooling2D((2, 2), name=f'pool{i + 1}')(x)

            x = layers.Dropout(0.2, name=f'dropout{i + 1}')(x)

        # Classification Head
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        x = layers.Dense(cfg['units'], activation='relu', name='fc1')(x)
        x = layers.Dropout(0.3, name='dropout_fc')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)

        model = models.Model(inputs, outputs, name=f'micro_cnn_{model_size}')
        return model


class UnifiedAudioDataLoader:
    """
    Robust Data Loader for SynTTS.
    Handles both Chinese (folder-based) and English (variant-based) label extraction.
    """

    def __init__(self, base_dir, sample_rate=16000):
        self.base_dir = base_dir
        self.sample_rate = sample_rate
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.preloaded_data = {}

        # English command mapping (for robustness)
        self.english_variants = {
            "play": "Play", "play_music": "Play", "start_play": "Play",
            "pause": "Pause", "stop": "Pause",
            "volume_up": "Volume up", "increase_volume": "Volume up",
            "volume_down": "Volume down", "decrease_volume": "Volume down",
            # Add other variants as needed...
        }

        # Chinese specific mappings
        self.chinese_variants = {
            'Hello_小智': 'Hello 小智',
            '嗨_三星小贝': '嗨 三星小贝',
            'Hello_XiaoZhi': 'Hello 小智'
        }

    def normalize_path(self, path):
        return str(Path(path).as_posix())

    def _extract_label(self, file_path):
        """Extracts label from path, handling both EN and ZH patterns."""
        file_path = self.normalize_path(file_path)
        parts = file_path.split('/')

        if len(parts) < 2: return "unknown"

        # Strategy: The label is usually the immediate parent folder of the wav file,
        # or the folder inside the subset folder.
        # Structure: Dataset/Subset/Label/File.wav -> index 1 (relative to subset) or 2 (relative to root)
        # Assuming input 'file_path' is relative to 'base_dir' (e.g., "Free_ST_Chinese/Label/file.wav")
        folder_name = parts[1]

        # 1. Check Chinese variants
        if folder_name in self.chinese_variants:
            return self.chinese_variants[folder_name]

        # 2. Check English variants (simple clean)
        folder_clean = folder_name.lower().replace('_', ' ').replace('-', ' ')

        # Heuristic: If it looks like an English command path, try to normalize
        # (This is simplified. For strict training, use the exact EnglishLoader logic if separated)
        # But since we pass 'language' arg, we could specialize.
        # Here we keep it generic:

        # Return raw folder name if no specific mapping found (usually correct for SynTTS)
        # Cleanup: remove file extensions from folder names if present
        if '.' in folder_name:
            folder_name = folder_name.split('.')[0]

        return folder_name

    def create_label_mapping(self, labels):
        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        print(f"[Info] Label mapping created: {len(unique_labels)} classes")
        return self.label_to_idx

    def preload_data(self, list_paths):
        print("[Info] Starting data preloading...")
        all_paths = []
        all_labels = []

        for p in list_paths:
            if not os.path.exists(p):
                print(f"[Warning] List not found: {p}")
                continue
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    rel_path = line.strip()
                    full_path = os.path.join(self.base_dir, rel_path)
                    if os.path.exists(full_path):
                        all_paths.append(full_path)
                        all_labels.append(self._extract_label(rel_path))

        self.create_label_mapping(all_labels)

        spectrograms = []
        indices = []
        # Store paths to allow subset filtering later
        valid_paths = []

        print(f"[Info] Loading {len(all_paths)} files...")
        for i, path in enumerate(all_paths):
            if i % 2000 == 0: print(f"       Progress: {i}/{len(all_paths)}")
            try:
                audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
                spec = self._audio_to_mel(audio)

                spectrograms.append(spec)
                indices.append(self.label_to_idx[all_labels[i]])
                valid_paths.append(self.normalize_path(path))
            except Exception as e:
                continue

        self.preloaded_data['spectrograms'] = np.array(spectrograms)
        self.preloaded_data['labels'] = np.array(indices)
        self.preloaded_data['paths'] = valid_paths

        print(f"[Info] Preloading done. Valid samples: {len(spectrograms)}")
        return self.preloaded_data

    def create_dataset(self, split_list, batch_size):
        target_paths = set()
        with open(split_list, 'r', encoding='utf-8') as f:
            for line in f:
                target_paths.add(self.normalize_path(os.path.join(self.base_dir, line.strip())))

        # Map loaded paths to indices
        # Build map for O(1) access
        path_to_idx = {p: i for i, p in enumerate(self.preloaded_data['paths'])}

        indices = []
        for tp in target_paths:
            if tp in path_to_idx:
                indices.append(path_to_idx[tp])

        print(f"[Info] Creating dataset from {os.path.basename(split_list)}: {len(indices)} samples")

        specs = self.preloaded_data['spectrograms'][indices]
        lbls = self.preloaded_data['labels'][indices]

        ds = tf.data.Dataset.from_tensor_slices((specs, lbls))
        ds = ds.shuffle(len(indices)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def _audio_to_mel(self, audio):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=40, hop_length=256, win_length=1024, fmin=20, fmax=4000
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        target = 49
        if log_mel.shape[1] < target:
            pad = target - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode='constant')
        else:
            log_mel = log_mel[:, :target]

        return np.expand_dims(log_mel.T, -1).astype(np.float32)


class ModelTrainer:
    def __init__(self, model, output_dir, model_name):
        self.model = model
        self.output_dir = output_dir
        self.model_name = model_name
        self.history = None
        os.makedirs(output_dir, exist_ok=True)

    def compile(self, lr):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, train_ds, val_ds, epochs):
        cbs = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.output_dir, f'best_{self.model_name}.keras'),
                save_best_only=True, monitor='val_accuracy', mode='max', verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
        ]
        self.history = self.model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=cbs, verbose=1)
        self._plot_history()

    def evaluate(self, test_ds):
        print("[Info] Evaluating...")
        res = self.model.evaluate(test_ds, verbose=1)
        return dict(zip(self.model.metrics_names, res))

    def _plot_history(self):
        if not self.history: return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history.history['accuracy'], label='Train')
        ax1.plot(self.history.history['val_accuracy'], label='Val')
        ax1.set_title('Accuracy');
        ax1.legend()
        ax2.plot(self.history.history['loss'], label='Train')
        ax2.plot(self.history.history['val_loss'], label='Val')
        ax2.set_title('Loss');
        ax2.legend()
        plt.savefig(os.path.join(self.output_dir, f'{self.model_name}_history.png'))
        plt.close()

    def save_report(self, res, args):
        with open(os.path.join(self.output_dir, 'report.json'), 'w') as f:
            json.dump({'config': vars(args), 'results': res}, f, indent=2)


def get_args():
    parser = argparse.ArgumentParser(description="Train MicroCNN on SynTTS")

    parser.add_argument('--dataset_root', type=str, default='./SynTTS-Commands-Media-Dataset')
    parser.add_argument('--splits_dir', type=str, default='./SynTTS-Commands-Media-Dataset/splits_by_language')
    parser.add_argument('--output_dir', type=str, default='./results/microcnn')
    parser.add_argument('--language', type=str, default='chinese', choices=['chinese', 'english'])

    parser.add_argument('--model_size', type=str, default='micro', choices=['nano', 'micro', 'small'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)  # Larger batch for small models
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--sample_rate', type=int, default=16000)

    return parser.parse_args()


def main():
    args = get_args()
    print(f"--- Config: MicroCNN ({args.model_size}) on {args.language} ---")
    print(vars(args))

    # 1. Loader
    loader = UnifiedAudioDataLoader(args.dataset_root, args.sample_rate)

    train_list = os.path.join(args.splits_dir, f'train_list_{args.language}.txt')
    val_list = os.path.join(args.splits_dir, f'validation_list_{args.language}.txt')
    test_list = os.path.join(args.splits_dir, f'test_list_{args.language}.txt')

    loader.preload_data([train_list, val_list, test_list])

    # 2. Datasets
    train_ds = loader.create_dataset(train_list, args.batch_size)
    val_ds = loader.create_dataset(val_list, args.batch_size)
    test_ds = loader.create_dataset(test_list, args.batch_size)

    # 3. Model
    builder = MicroCNNBuilder((49, 40, 1), len(loader.label_to_idx))
    model = builder.build_micro_cnn(args.model_size)
    model.summary()

    # Check params
    total_params = model.count_params()
    if total_params < 10000:
        print("[Info] This is an Ultra-lightweight model!")

    # 4. Train
    trainer = ModelTrainer(model, args.output_dir, f'microcnn_{args.model_size}_{args.language}')
    trainer.compile(args.lr)
    trainer.train(train_ds, val_ds, args.epochs)

    # 5. Eval
    res = trainer.evaluate(test_ds)
    print(f"\n[Result] Test Acc: {res['accuracy']:.4f}")
    trainer.save_report(res, args)


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    main()