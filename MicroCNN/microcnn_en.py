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
        Builds MicroCNN model variants.
        Args:
            model_size: 'nano', 'micro', 'small'
        """
        inputs = tf.keras.Input(shape=self.input_shape)

        # Architecture Configurations
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
            # Depthwise Separable Conv (Depthwise + Pointwise)
            x = layers.DepthwiseConv2D(kernel_size, padding='same', name=f'dw_conv{i + 1}')(x)
            x = layers.BatchNormalization(name=f'bn_dw{i + 1}')(x)
            x = layers.ReLU(name=f'relu_dw{i + 1}')(x)

            x = layers.Conv2D(filters, (1, 1), padding='same', name=f'pw_conv{i + 1}')(x)
            x = layers.BatchNormalization(name=f'bn_pw{i + 1}')(x)
            x = layers.ReLU(name=f'relu_pw{i + 1}')(x)

            # Pooling (skip for last layer to keep spatial dims for GAP)
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


class EnglishAudioDataLoader:
    """Data Loader specific for English Subset."""

    def __init__(self, base_dir, sample_rate=16000):
        self.base_dir = base_dir
        self.sample_rate = sample_rate
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.preloaded_data = {}

        # Standard 23 English Commands
        self.english_commands = [
            # Playback Control
            "Play", "Pause", "Resume", "Play from start", "Repeat song",
            # Navigation
            "Previous track", "Next track", "Last song", "Skip song", "Jump to first track",
            # Volume Control
            "Volume up", "Volume down", "Mute", "Set volume to 50%", "Max volume",
            # Communication
            "Answer call", "Hang up", "Decline call",
            # Wake Words
            "Hey Siri", "OK Google", "Hey Google", "Alexa", "Hi Bixby"
        ]

    def normalize_path(self, path):
        return str(Path(path).as_posix())

    def _extract_english_label(self, file_path):
        """Extracts label by matching command names in the file path."""
        file_path = self.normalize_path(file_path)
        path_lower = file_path.lower()

        # 1. Direct Match against command list
        for command in self.english_commands:
            # Clean command: "Set volume to 50%" -> "set_volume_to_50"
            search_pattern = command.lower().replace(' ', '_').replace('50%', '50')
            if search_pattern in path_lower:
                return command

        # 2. Check Folder Name
        parts = file_path.split('/')
        if len(parts) >= 2:
            folder_name = parts[1]
            for command in self.english_commands:
                if command.lower().replace(' ', '_') in folder_name.lower():
                    return command

        return "unknown"

    def create_label_mapping(self):
        self.label_to_idx = {label: idx for idx, label in enumerate(self.english_commands)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        print(f"[Info] Label mapping created: {len(self.english_commands)} classes")
        return self.label_to_idx, self.idx_to_label

    def preload_all_english_data(self, file_list_paths):
        print("[Info] Starting English data preloading...")

        all_file_paths = []
        all_labels = []

        for file_list_path in file_list_paths:
            if not os.path.exists(file_list_path):
                print(f"[Warning] List file missing: {file_list_path}")
                continue

            with open(file_list_path, 'r', encoding='utf-8') as f:
                for line in f:
                    relative_path = line.strip()
                    full_path = os.path.join(self.base_dir, relative_path)
                    if os.path.exists(full_path):
                        all_file_paths.append(full_path)
                        label = self._extract_english_label(relative_path)
                        all_labels.append(label)

        self.create_label_mapping()

        print(f"[Info] Loading {len(all_file_paths)} audio files...")
        spectrograms = []
        labels_indices = []
        failed_count = 0

        for i, file_path in enumerate(all_file_paths):
            if i % 2000 == 0:
                print(f"       Progress: {i}/{len(all_file_paths)}")

            try:
                audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
                spectrogram = self._audio_to_mel_spectrogram(audio, sr)

                label = all_labels[i]
                if label in self.label_to_idx:
                    label_idx = self.label_to_idx[label]
                    spectrograms.append(spectrogram)
                    labels_indices.append(label_idx)
                else:
                    continue

            except Exception as e:
                failed_count += 1
                continue

        self.preloaded_data['spectrograms'] = np.array(spectrograms)
        self.preloaded_data['labels'] = np.array(labels_indices)
        self.preloaded_data['raw_paths'] = all_file_paths

        print(f"[Info] Preloading complete. Success: {len(spectrograms)}, Failed: {failed_count}")
        self._show_label_statistics(all_labels)
        return self.preloaded_data

    def _show_label_statistics(self, all_labels):
        label_counts = Counter(all_labels)
        print("\n--- Label Distribution ---")
        for label in self.english_commands:
            count = label_counts.get(label, 0)
            print(f"  {'✓' if count > 0 else '✗'} {label}: {count}")
        print("--------------------------")

    def create_english_dataset(self, split_type, splits_dir, batch_size=32):
        if not self.preloaded_data:
            raise ValueError("Data not preloaded.")

        file_list_path = os.path.join(splits_dir, f'{split_type}_list_english.txt')

        split_files = set()
        with open(file_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                full_path = os.path.join(self.base_dir, line.strip())
                split_files.add(self.normalize_path(full_path))

        indices = []
        # Basic matching logic (assumes no load failures for index alignment)
        # For full robustness, a path->index map is preferred.
        valid_idx = 0
        for i, p in enumerate(self.preloaded_data['raw_paths']):
            if valid_idx < len(self.preloaded_data['labels']):
                if self.normalize_path(p) in split_files:
                    indices.append(valid_idx)
                valid_idx += 1

        print(f"[Info] {split_type} set: {len(indices)} samples")

        split_spectrograms = self.preloaded_data['spectrograms'][indices]
        split_labels = self.preloaded_data['labels'][indices]

        dataset = tf.data.Dataset.from_tensor_slices((split_spectrograms, split_labels))
        if split_type == 'train':
            dataset = dataset.shuffle(buffer_size=len(indices))

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset, len(indices)

    def _audio_to_mel_spectrogram(self, audio, sample_rate):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_mels=40, hop_length=256, win_length=1024, fmin=20, fmax=4000
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        target_width = 49
        if log_mel_spec.shape[1] < target_width:
            pad_width = target_width - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            log_mel_spec = log_mel_spec[:, :target_width]

        log_mel_spec = log_mel_spec.T
        log_mel_spec = np.expand_dims(log_mel_spec, -1)
        return log_mel_spec.astype(np.float32)


class ModelTrainer:
    def __init__(self, model, output_dir, model_name):
        self.model = model
        self.output_dir = output_dir
        self.model_name = model_name
        self.history = None
        os.makedirs(output_dir, exist_ok=True)

    def compile_model(self, learning_rate):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, train_dataset, val_dataset, epochs):
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.output_dir, f'best_{self.model_name}.keras'),
                save_best_only=True, monitor='val_accuracy', mode='max', verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
        ]

        print(f"[Info] Starting training for {self.model_name}...")
        self.history = self.model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks,
                                      verbose=1)
        self.plot_history()
        return self.history

    def evaluate(self, test_ds):
        print("[Info] Evaluating model...")
        results = self.model.evaluate(test_ds, verbose=1)
        return dict(zip(self.model.metrics_names, results))

    def plot_history(self):
        if not self.history: return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.history.history['accuracy'], label='Train')
        ax1.plot(self.history.history['val_accuracy'], label='Val')
        ax1.set_title('Accuracy')
        ax1.legend()

        ax2.plot(self.history.history['loss'], label='Train')
        ax2.plot(self.history.history['val_loss'], label='Val')
        ax2.set_title('Loss')
        ax2.legend()

        plt.savefig(os.path.join(self.output_dir, f'{self.model_name}_history.png'))
        plt.close()

    def save_report(self, test_results, config):
        report = {
            'model_name': self.model_name,
            'config': config,
            'test_results': test_results
        }
        with open(os.path.join(self.output_dir, f'{self.model_name}_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

    def print_model_summary(self):
        total_params = self.model.count_params()
        print("\n--- Model Stats ---")
        print(f"Total Params: {total_params:,}")
        if total_params < 10000:
            print("✅ Ultra-lightweight (< 10K)")
        elif total_params < 50000:
            print("✅ Lightweight (< 50K)")
        print("-------------------")


def get_args():
    parser = argparse.ArgumentParser(description="Train MicroCNN on SynTTS (English)")

    parser.add_argument('--dataset_root', type=str, default='./SynTTS-Commands-Media-Dataset', help='Dataset root')
    parser.add_argument('--splits_dir', type=str, default='./SynTTS-Commands-Media-Dataset/splits_by_language',
                        help='Splits dir')
    parser.add_argument('--output_dir', type=str, default='./results/microcnn_english', help='Output dir')

    parser.add_argument('--model_size', type=str, default='micro', choices=['nano', 'micro', 'small'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--sample_rate', type=int, default=16000)

    return parser.parse_args()


def main():
    args = get_args()
    print(f"--- Config: MicroCNN ({args.model_size}) English ---")
    print(vars(args))

    # 1. Loader
    data_loader = EnglishAudioDataLoader(args.dataset_root, sample_rate=args.sample_rate)

    # 2. Preload
    lists = [
        os.path.join(args.splits_dir, 'train_list_english.txt'),
        os.path.join(args.splits_dir, 'validation_list_english.txt'),
        os.path.join(args.splits_dir, 'test_list_english.txt')
    ]
    data_loader.preload_all_english_data(lists)

    # 3. Create Datasets
    train_ds, _ = data_loader.create_english_dataset('train', args.splits_dir, args.batch_size)
    val_ds, _ = data_loader.create_english_dataset('validation', args.splits_dir, args.batch_size)
    test_ds, _ = data_loader.create_english_dataset('test', args.splits_dir, args.batch_size)

    # 4. Build Model
    builder = MicroCNNBuilder((49, 40, 1), num_classes=23)
    model = builder.build_micro_cnn(args.model_size)
    model.summary()

    # 5. Train
    trainer = ModelTrainer(model, args.output_dir, f'microcnn_{args.model_size}_en')
    trainer.compile_model(args.lr)
    trainer.print_model_summary()

    trainer.train(train_ds, val_ds, args.epochs)

    # 6. Evaluate
    res = trainer.evaluate(test_ds)
    print(f"\n[Result] Test Acc: {res['accuracy']:.4f}")

    trainer.save_report(res, vars(args))


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    main()