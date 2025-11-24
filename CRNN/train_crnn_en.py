import tensorflow as tf
import numpy as np
import os
import json
import pandas as pd
from tensorflow.keras import layers, models
import matplotlib

matplotlib.use('Agg')  # Set backend before importing pyplot
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


class CRNNBuilder:
    """CRNN Model Builder - CNN for feature extraction + RNN for temporal modeling."""

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape  # (time_steps, n_mels, 1)
        self.num_classes = num_classes

    def build_crnn_model(self, cnn_filters=[32, 64, 128], rnn_units=128, dropout_rate=0.3):
        inputs = tf.keras.Input(shape=self.input_shape)

        # CNN Part - Feature Extraction
        x = inputs

        # Block 1
        x = layers.Conv2D(cnn_filters[0], (3, 3), padding='same', name='conv1')(x)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.ReLU(name='relu1')(x)
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)
        x = layers.Dropout(dropout_rate, name='dropout1')(x)

        # Block 2
        x = layers.Conv2D(cnn_filters[1], (3, 3), padding='same', name='conv2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.ReLU(name='relu2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)
        x = layers.Dropout(dropout_rate, name='dropout2')(x)

        # Block 3
        x = layers.Conv2D(cnn_filters[2], (3, 3), padding='same', name='conv3')(x)
        x = layers.BatchNormalization(name='bn3')(x)
        x = layers.ReLU(name='relu3')(x)
        x = layers.MaxPooling2D((1, 2), name='pool3')(x)  # Pool only freq dimension
        x = layers.Dropout(dropout_rate, name='dropout3')(x)

        # Reshape for RNN
        cnn_output_shape = x.shape
        # print(f"[Debug] CNN Output Shape: {cnn_output_shape}")

        # Reshape: (batch, time, height * width * channels)
        x = layers.Reshape((cnn_output_shape[1], cnn_output_shape[2] * cnn_output_shape[3]),
                           name='reshape_to_rnn')(x)

        # RNN Part - Temporal Modeling
        x = layers.Bidirectional(
            layers.GRU(rnn_units, return_sequences=True, dropout=dropout_rate),
            name='bgru1'
        )(x)

        x = layers.Bidirectional(
            layers.GRU(rnn_units, return_sequences=False, dropout=dropout_rate),
            name='bgru2'
        )(x)

        # Classification Head
        x = layers.Dense(256, activation='relu', name='fc1')(x)
        x = layers.Dropout(dropout_rate, name='dropout_fc1')(x)
        x = layers.Dense(128, activation='relu', name='fc2')(x)
        x = layers.Dropout(dropout_rate, name='dropout_fc2')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)

        model = models.Model(inputs, outputs, name='crnn_english')
        return model

    def build_light_crnn(self):
        return self.build_crnn_model(cnn_filters=[16, 32, 64], rnn_units=64, dropout_rate=0.2)

    def build_heavy_crnn(self):
        return self.build_crnn_model(cnn_filters=[64, 128, 256], rnn_units=256, dropout_rate=0.4)


class EnglishAudioDataLoader:
    """Data Loader specifically for the English subset."""

    def __init__(self, base_dir, sample_rate=16000):
        self.base_dir = base_dir
        self.sample_rate = sample_rate
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.preloaded_data = {}

        # Define 23 English commands as per the dataset
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
        """Robust label extraction from file path."""
        file_path = self.normalize_path(file_path)

        # 1. Try matching standard commands in path
        for command in self.english_commands:
            # Handle spaces and special chars (e.g., "50%")
            search_pattern = command.lower().replace(' ', '_').replace('50%', '50')
            if search_pattern in file_path.lower():
                return command

        # 2. Fallback: Check folder name directly
        parts = file_path.split('/')
        if len(parts) >= 2:
            folder_name = parts[1]  # Assuming structure: dataset/label/file.wav
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
        print("[Info] Preloading English data into memory...")

        all_file_paths = []
        all_labels = []

        # Collect paths
        for file_list_path in file_list_paths:
            if not os.path.exists(file_list_path):
                print(f"[Warning] List file not found: {file_list_path}")
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
                    print(f"[Warning] Unknown label '{label}' for {file_path}")
                    # Skip unknown labels to avoid polluting dataset
                    continue

            except Exception as e:
                # print(f"[Error] Failed to load {file_path}: {e}")
                failed_count += 1
                continue

        self.preloaded_data['spectrograms'] = np.array(spectrograms)
        self.preloaded_data['labels'] = np.array(labels_indices)
        # Store valid file paths only
        self.preloaded_data['file_paths'] = [
                                                all_file_paths[i] for i in range(len(all_file_paths))
                                                if i < len(all_labels) and all_labels[i] in self.label_to_idx
                                            ][:len(spectrograms)]  # Ensure length match

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

        # Identify files belonging to this split
        split_files = set()
        with open(file_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                full_path = os.path.join(self.base_dir, line.strip())
                split_files.add(self.normalize_path(full_path))

        # Map preloaded indices
        indices = []
        # Optimization: Map path to index for O(1) lookup
        path_to_idx = {self.normalize_path(p): i for i, p in enumerate(self.preloaded_data['file_paths'])}

        for p in split_files:
            p_norm = self.normalize_path(p)
            if p_norm in path_to_idx:
                indices.append(path_to_idx[p_norm])

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

        # Pad/Crop to fixed width (49 frames)
        target_width = 49
        if log_mel_spec.shape[1] < target_width:
            pad_width = target_width - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            log_mel_spec = log_mel_spec[:, :target_width]

        log_mel_spec = log_mel_spec.T
        log_mel_spec = np.expand_dims(log_mel_spec, -1)
        return log_mel_spec.astype(np.float32)


class CRNNTrainer:
    def __init__(self, model, output_dir, model_name='crnn_english'):
        self.model = model
        self.output_dir = output_dir
        self.model_name = model_name
        self.history = None
        os.makedirs(output_dir, exist_ok=True)

    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, train_ds, val_ds, epochs=50):
        print(f"[Info] Starting training for {self.model_name}...")

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.output_dir, f'best_{self.model_name}.keras'),
                save_best_only=True, monitor='val_accuracy', mode='max', verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
        ]

        self.history = self.model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks, verbose=1)
        self.plot_training_history()
        return self.history

    def evaluate(self, test_ds):
        print("[Info] Evaluating model...")
        results = self.model.evaluate(test_ds, verbose=1)
        return dict(zip(self.model.metrics_names, results))

    def plot_training_history(self):
        if self.history is None: return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Train Acc', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Val Acc', linewidth=2)
        ax1.set_title(f'{self.model_name} Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Loss
        ax2.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        ax2.set_title(f'{self.model_name} Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        save_path = os.path.join(self.output_dir, f'{self.model_name}_history.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[Info] Training plot saved to {save_path}")

    def save_report(self, test_results, config):
        report = {
            'model_name': self.model_name,
            'config': config,
            'test_results': test_results,
            'history_summary': {
                'final_acc': self.history.history['accuracy'][-1],
                'final_val_acc': self.history.history['val_accuracy'][-1],
                'epochs_trained': len(self.history.history['accuracy'])
            }
        }
        with open(os.path.join(self.output_dir, f'{self.model_name}_report.json'), 'w') as f:
            json.dump(report, f, indent=2)


def get_args():
    parser = argparse.ArgumentParser(description="Train CRNN on SynTTS-Commands (English Subset)")

    # Paths
    parser.add_argument('--dataset_root', type=str, default='./SynTTS-Commands-Media-Dataset',
                        help='Root directory of the dataset')
    parser.add_argument('--splits_dir', type=str, default='./SynTTS-Commands-Media-Dataset/splits_by_language',
                        help='Directory containing list files')
    parser.add_argument('--output_dir', type=str, default='./results/crnn_english',
                        help='Directory to save outputs')

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Sampling rate (16000 is standard for KWS)')

    # Model Config
    parser.add_argument('--model_variant', type=str, default='standard',
                        choices=['light', 'standard', 'heavy'])

    return parser.parse_args()


def main():
    args = get_args()
    print("=" * 60)
    print(f" SynTTS English KWS Benchmark - CRNN ({args.model_variant})")
    print("=" * 60)
    print(f"Config: {vars(args)}")

    # 1. Initialize Loader
    data_loader = EnglishAudioDataLoader(args.dataset_root, sample_rate=args.sample_rate)

    # 2. Preload Data
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
    builder = CRNNBuilder((49, 40, 1), num_classes=23)
    if args.model_variant == 'light':
        model = builder.build_light_crnn()
    elif args.model_variant == 'heavy':
        model = builder.build_heavy_crnn()
    else:
        model = builder.build_crnn_model()

    model.summary()

    # 5. Train
    trainer = CRNNTrainer(model, args.output_dir, f'crnn_{args.model_variant}_en')
    trainer.compile_model(args.lr)
    trainer.train(train_ds, val_ds, args.epochs)

    # 6. Evaluate
    test_results = trainer.evaluate(test_ds)
    print(f"\n[Result] Test Accuracy: {test_results['accuracy']:.4f}")

    # 7. Save Report
    trainer.save_report(test_results, vars(args))
    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    # Enable GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[Info] GPU Enabled: {gpus[0]}")
        except RuntimeError as e:
            print(e)

    main()