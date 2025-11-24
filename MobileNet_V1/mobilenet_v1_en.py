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


class AudioMobileNetBuilder:
    """
    Audio-MobileNet Builder based on MobileNet V1 architecture.
    Uses Depthwise Separable Convolutions and Width Multiplier (alpha).
    """

    def __init__(self, input_shape, num_classes, alpha=1.0):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.alpha = alpha  # Width Multiplier (e.g., 0.25, 0.5, 0.75, 1.0)

    def _depthwise_separable_block(self, x, filters, stride=1, block_name=""):
        """Standard MobileNet V1 Depthwise Separable Block."""
        # 1. Depthwise
        x = layers.DepthwiseConv2D(
            (3, 3),
            strides=(stride, stride),
            padding='same',
            name=f'{block_name}_depthwise'
        )(x)
        x = layers.BatchNormalization(name=f'{block_name}_depthwise_bn')(x)
        x = layers.ReLU(max_value=6.0, name=f'{block_name}_depthwise_relu6')(x)

        # 2. Pointwise
        # Apply alpha width multiplier to pointwise filters
        pointwise_filters = int(filters * self.alpha)
        x = layers.Conv2D(
            pointwise_filters,
            (1, 1),
            padding='same',
            name=f'{block_name}_pointwise'
        )(x)
        x = layers.BatchNormalization(name=f'{block_name}_pointwise_bn')(x)
        x = layers.ReLU(max_value=6.0, name=f'{block_name}_pointwise_relu6')(x)

        return x

    def build_audio_mobilenet(self):
        """Builds the Audio-MobileNet model."""
        inputs = tf.keras.Input(shape=self.input_shape)

        # Initial Conv Layer
        first_filters = int(32 * self.alpha)
        x = layers.Conv2D(
            first_filters,
            (3, 3),
            strides=(2, 2),
            padding='same',
            name='conv1'
        )(inputs)
        x = layers.BatchNormalization(name='bn_conv1')(x)
        x = layers.ReLU(max_value=6.0, name='relu6_conv1')(x)

        # MobileNet V1 Block Configuration
        # (filters, strides, block_name)
        block_config = [
            (64, 1, "block1"),
            (128, 2, "block2"),
            (128, 1, "block3"),
            (256, 2, "block4"),
            (256, 1, "block5"),
            (512, 2, "block6"),
            (512, 1, "block7"),
            (512, 1, "block8"),
            (512, 1, "block9"),
            (512, 1, "block10"),
            (512, 1, "block11"),
            (1024, 2, "block12"),
            (1024, 1, "block13")
        ]

        for filters, stride, block_name in block_config:
            x = self._depthwise_separable_block(x, filters, stride, block_name)
            # Add dropout for regularization in audio tasks
            x = layers.Dropout(0.2, name=f'{block_name}_dropout')(x)

        # Classification Head
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)

        # Optional: Dense layer before softmax (can be adjusted based on model size)
        fc_units = int(1024 * self.alpha)
        x = layers.Dense(fc_units, activation='relu', name='fc1')(x)
        x = layers.Dropout(0.3, name='dropout_fc')(x)

        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)

        model = models.Model(inputs, outputs, name=f'mobilenet_v1_alpha{self.alpha}')
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
        """
        Extracts label by matching command names in the file path.
        Handles special characters like '%' and spaces.
        """
        file_path = self.normalize_path(file_path)
        path_lower = file_path.lower()

        # 1. Direct Match against command list
        for command in self.english_commands:
            # Clean command: "Set volume to 50%" -> "set_volume_to_50"
            search_pattern = command.lower().replace(' ', '_').replace('50%', '50')
            if search_pattern in path_lower:
                return command

        # 2. Fallback: Check Folder Name
        parts = file_path.split('/')
        if len(parts) >= 2:
            folder_name = parts[1]
            for command in self.english_commands:
                # Loose matching for folder names
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
        # Store paths for split filtering
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
        # Simplified matching (assumes index alignment if no failures)
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
        print(f"Model: {self.model_name}")
        print(f"Total Params: {total_params:,}")
        if total_params < 1000000:
            print("✅ Lightweight Model (< 1M)")
        print("-------------------")


def get_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV1 on SynTTS (English)")

    parser.add_argument('--dataset_root', type=str, default='./SynTTS-Commands-Media-Dataset', help='Dataset root')
    parser.add_argument('--splits_dir', type=str, default='./SynTTS-Commands-Media-Dataset/splits_by_language',
                        help='Splits dir')
    parser.add_argument('--output_dir', type=str, default='./results/mobilenet_english', help='Output dir')

    # MobileNet specific args
    parser.add_argument('--alpha', type=float, default=0.75, help='Width Multiplier (e.g., 0.25, 0.5, 0.75, 1.0)')

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--sample_rate', type=int, default=16000)

    return parser.parse_args()


def main():
    args = get_args()
    print(f"--- Config: MobileNetV1 (alpha={args.alpha}) English ---")
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
    # Input: (Time, Freq, 1) -> (49, 40, 1)
    builder = AudioMobileNetBuilder((49, 40, 1), num_classes=23, alpha=args.alpha)
    model = builder.build_audio_mobilenet()
    model.summary()

    # 5. Train
    model_name = f'mobilenet_a{args.alpha}_en'
    trainer = ModelTrainer(model, args.output_dir, model_name)
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