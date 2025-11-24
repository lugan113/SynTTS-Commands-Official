import tensorflow as tf
import numpy as np
import os
import json
import pandas as pd
from tensorflow.keras import layers, models
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
    """CRNN Model Builder - Combining CNN for feature extraction and RNN for temporal modeling."""

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
        # print(f"DEBUG: CNN Output Shape: {cnn_output_shape}")

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

        model = models.Model(inputs, outputs, name='crnn_kws')
        return model


class AudioDataLoader:
    """Handles data loading, preprocessing, and label mapping."""

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.preloaded_data = {}

    def create_label_mapping(self, labels):
        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        print(f"[Info] Label mapping created: {len(unique_labels)} classes")
        return self.label_to_idx, self.idx_to_label

    def normalize_path(self, path):
        return str(Path(path).as_posix())

    def _contains_chinese(self, text):
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        return bool(chinese_pattern.search(text))

    def _extract_label_from_path(self, file_path):
        """Extracts label from folder structure specific to SynTTS dataset."""
        file_path = self.normalize_path(file_path)
        parts = file_path.split('/')

        if len(parts) < 2:
            return "unknown"

        folder_name = parts[1]  # Assuming data/subset/label/file.wav

        # Mapping for specific folders in SynTTS dataset
        wake_word_mapping = {
            'Hello_小智': 'Hello 小智',
            '嗨_三星小贝': '嗨 三星小贝',
        }

        if folder_name in wake_word_mapping:
            return wake_word_mapping[folder_name]

        if '_' in folder_name:
            folder_parts = folder_name.split('_')
            if self._contains_chinese(folder_parts[-1]):
                folder_name = folder_parts[-1]

        if '.' in folder_name:
            folder_name = folder_name.split('.')[0]

        return folder_name

    def preload_all_data(self, file_list_paths):
        """Preloads all audio data into memory (RAM intensive)."""
        print("[Info] Starting data preloading...")

        all_file_paths = []
        all_labels = []

        # 1. Collect all paths
        for file_list_path in file_list_paths:
            if not os.path.exists(file_list_path):
                print(f"[Warning] List file not found: {file_list_path}")
                continue

            print(f"[Info] Reading list: {file_list_path}")
            with open(file_list_path, 'r', encoding='utf-8') as f:
                for line in f:
                    relative_path = line.strip()
                    full_path = os.path.join(self.base_dir, relative_path)
                    if os.path.exists(full_path):
                        all_file_paths.append(full_path)
                        label = self._extract_label_from_path(relative_path)
                        all_labels.append(label)

        if not all_file_paths:
            raise ValueError("No valid files found. Please check dataset path and list files.")

        self.create_label_mapping(all_labels)

        # 2. Load Audio
        print(f"[Info] Loading {len(all_file_paths)} audio files...")
        total_files = len(all_file_paths)
        spectrograms = []
        labels_indices = []
        failed_count = 0

        for i, file_path in enumerate(all_file_paths):
            if i % 2000 == 0:
                print(f"       Progress: {i}/{total_files} ({i / total_files * 100:.1f}%)")

            try:
                audio, sr = librosa.load(file_path, sr=16000, mono=True)  # Changed default to 16k for KWS
                spectrogram = self._audio_to_mel_spectrogram_librosa(audio, sr)
                spectrograms.append(spectrogram)
                labels_indices.append(self.label_to_idx[all_labels[i]])
            except Exception as e:
                # print(f"[Error] Failed to load {file_path}: {e}")
                failed_count += 1
                # Add placeholder to maintain sync (or skip) - here we skip
                continue

        self.preloaded_data['spectrograms'] = np.array(spectrograms)
        self.preloaded_data['labels'] = np.array(labels_indices)
        self.preloaded_data['file_paths'] = all_file_paths  # Note: this might be longer than specs if skipped

        print(f"[Info] Preloading complete. Success: {len(spectrograms)}, Failed: {failed_count}")
        return self.preloaded_data

    def create_dataset_from_preloaded(self, split_list_path, batch_size=32):
        if not self.preloaded_data:
            raise ValueError("Data not preloaded.")

        # Read specific split list
        split_files = set()
        with open(split_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                full_path = os.path.join(self.base_dir, line.strip())
                split_files.add(self.normalize_path(full_path))

        # Filter indices
        indices = []
        # Create a map for faster lookup if needed, but simple loop is okay for <1M files
        # Note: self.preloaded_data['file_paths'] might contain paths not in this split
        # We need to map current loaded paths to indices

        # Optimization: Map loaded path to index
        path_to_idx = {self.normalize_path(p): i for i, p in enumerate(self.preloaded_data['file_paths']) if
                       i < len(self.preloaded_data['labels'])}

        valid_indices = []
        for p in split_files:
            p_norm = self.normalize_path(p)
            if p_norm in path_to_idx:
                valid_indices.append(path_to_idx[p_norm])

        print(f"[Info] Created dataset from {os.path.basename(split_list_path)}: {len(valid_indices)} samples")

        split_spectrograms = self.preloaded_data['spectrograms'][valid_indices]
        split_labels = self.preloaded_data['labels'][valid_indices]

        dataset = tf.data.Dataset.from_tensor_slices((split_spectrograms, split_labels))
        dataset = dataset.shuffle(len(valid_indices)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset, len(valid_indices)

    def _audio_to_mel_spectrogram_librosa(self, audio, sample_rate):
        # Config matches standard KWS benchmarks (e.g., Google Speech Commands)
        # You may verify these params match your generated data
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_mels=40, hop_length=256, win_length=1024, fmin=20, fmax=4000
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Pad or Crop to fixed width (e.g. 49 frames for ~1 second at 16k sr)
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
    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def compile_model(self, learning_rate):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, train_ds, val_ds, epochs):
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.output_dir, 'best_model.keras'),
                save_best_only=True, monitor='val_accuracy', mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        ]

        history = self.model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks, verbose=1)
        self.plot_history(history)
        return history

    def evaluate(self, test_ds):
        results = self.model.evaluate(test_ds, verbose=0)
        return dict(zip(self.model.metrics_names, results))

    def plot_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Val')
        ax1.set_title('Accuracy')
        ax1.legend()
        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Val')
        ax2.set_title('Loss')
        ax2.legend()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'))
        plt.close()


def get_args():
    parser = argparse.ArgumentParser(description="Train CRNN on SynTTS-Commands Dataset")

    # Path Arguments
    parser.add_argument('--dataset_root', type=str, default='./SynTTS-Commands-Media-Dataset',
                        help='Root directory of the dataset containing wav files')
    parser.add_argument('--splits_dir', type=str, default='./SynTTS-Commands-Media-Dataset/splits_by_language',
                        help='Directory containing train/val/test list files')
    parser.add_argument('--output_dir', type=str, default='./results/crnn',
                        help='Directory to save models and logs')
    parser.add_argument('--language', type=str, default='chinese', choices=['chinese', 'english'],
                        help='Target subset language')

    # Model Arguments
    parser.add_argument('--model_size', type=str, default='standard', choices=['light', 'standard', 'heavy'])

    # Training Arguments
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)

    return parser.parse_args()


def main():
    args = get_args()
    print(f"--- Configuration ---")
    print(vars(args))
    print(f"---------------------")

    # Paths construction
    train_list = os.path.join(args.splits_dir, f'train_list_{args.language}.txt')
    val_list = os.path.join(args.splits_dir, f'validation_list_{args.language}.txt')
    test_list = os.path.join(args.splits_dir, f'test_list_{args.language}.txt')

    # 1. Data Loading
    data_loader = AudioDataLoader(args.dataset_root)
    # Preload everything referenced in the lists
    data_loader.preload_all_data([train_list, val_list, test_list])

    # 2. Create Datasets
    train_ds, train_len = data_loader.create_dataset_from_preloaded(train_list, args.batch_size)
    val_ds, val_len = data_loader.create_dataset_from_preloaded(val_list, args.batch_size)
    test_ds, test_len = data_loader.create_dataset_from_preloaded(test_list, args.batch_size)

    num_classes = len(data_loader.label_to_idx)
    print(f"[Info] Num Classes: {num_classes}")

    # 3. Model Building
    builder = CRNNBuilder((49, 40, 1), num_classes)

    if args.model_size == 'light':
        model = builder.build_crnn_model(cnn_filters=[16, 32, 64], rnn_units=64, dropout_rate=0.2)
    elif args.model_size == 'heavy':
        model = builder.build_crnn_model(cnn_filters=[64, 128, 256], rnn_units=256, dropout_rate=0.4)
    else:
        model = builder.build_crnn_model()  # Standard

    model.summary()

    # 4. Training
    trainer = ModelTrainer(model, args.output_dir)
    trainer.compile_model(args.lr)
    trainer.train(train_ds, val_ds, args.epochs)

    # 5. Evaluation
    test_results = trainer.evaluate(test_ds)
    print(f"\n[Result] Test Set Evaluation: {test_results}")

    # Save final metadata
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump({'config': vars(args), 'results': test_results}, f, indent=2)


if __name__ == "__main__":
    # GPU Memory Growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    main()