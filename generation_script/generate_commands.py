import os
import sys
import argparse
import multiprocessing
import torch
import torchaudio
from tqdm import tqdm
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# Add third_party path if necessary
# sys.path.append('third_party/Matcha-TTS') 

def safe_filename(text):
    """
    Sanitize the filename.
    Replaces spaces with underscores and removes invalid characters.
    Example: "Answer call" -> "Answer_call"
    """
    # Replace spaces with underscores first
    text = text.replace(" ", "_")
    # Keep only alphanumeric, underscores, hyphens, and dots
    return "".join(c if c.isalnum() or c in "._-" else "" for c in text)

class CosyVoiceWorker:
    """Worker class wrapping CosyVoice model instances."""

    def __init__(self, gpu_id, num_instances=1, model_path='iic/CosyVoice2-0.5B'):
        self.gpu_id = gpu_id
        self.num_instances = num_instances

        # Determine device
        if torch.cuda.is_available() and gpu_id >= 0 and gpu_id < torch.cuda.device_count():
            self.device = f"cuda:{gpu_id}"
            print(f"[Worker] Using GPU {gpu_id} with {num_instances} model instances")
        else:
            self.device = "cpu"
            print(f"[Worker] Using CPU with {num_instances} model instances")

        # Create model instances
        self.models = []
        for i in range(num_instances):
            try:
                model = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=True)
                
                # Move model to device explicitly
                if hasattr(model, 'model'):
                    model.model.to(self.device)
                if hasattr(model, 'vocoder') and model.vocoder is not None:
                    model.vocoder.to(self.device)
                
                self.models.append(model)
            except Exception as e:
                print(f"Error loading model instance {i}: {e}")

        self.model_index = 0
        self.lock = multiprocessing.Lock()

    def get_next_model(self):
        """Round-robin selection of model instances."""
        with self.lock:
            model = self.models[self.model_index]
            self.model_index = (self.model_index + 1) % self.num_instances
            return model

    def process_sample(self, args):
        """Process a single audio sample to generate commands."""
        sample_file, output_base_dir, command_lines, voice_samples_dir = args
        sample_name = os.path.splitext(sample_file)[0]
        sample_path = os.path.join(voice_samples_dir, sample_file)

        try:
            # Load prompt speech (limit to 25s)
            prompt_speech_16k = load_wav(sample_path, 16000, max_duration_sec=25)
            generated_count = 0

            for command in command_lines:
                safe_command = safe_filename(command)
                output_dir = os.path.join(output_base_dir, safe_command)
                
                # Ensure filename format: sampleName_commandName.wav
                output_filename = f"{sample_name}_{safe_command}.wav"
                output_path = os.path.join(output_dir, output_filename)

                if os.path.exists(output_path):
                    continue

                try:
                    model = self.get_next_model()
                    # Inference
                    # Note: CosyVoice2 inference_cross_lingual typically handles mixed language input well.
                    for i, j in enumerate(model.inference_cross_lingual(command, prompt_speech_16k, stream=False)):
                        speech_data = j['tts_speech'].cpu()
                        torchaudio.save(output_path, speech_data, model.sample_rate)
                        generated_count += 1
                except Exception as cmd_error:
                    # print(f"Failed to generate '{command}' for {sample_name}: {cmd_error}")
                    pass

            return True, sample_name, generated_count
        except Exception as sample_error:
            return False, sample_name, str(sample_error)

def worker_process(gpu_id, num_instances, model_path, task_queue, result_queue):
    """Main loop for worker process."""
    try:
        worker = CosyVoiceWorker(gpu_id, num_instances, model_path)
        while True:
            task = task_queue.get()
            if task is None:
                result_queue.put(None)
                break
            result = worker.process_sample(task)
            result_queue.put(result)
    except Exception as e:
        print(f"Worker process error: {e}")
        result_queue.put((False, "worker_error", str(e)))

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic KWS dataset using CosyVoice.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing source wav files (seeds).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated audio.")
    parser.add_argument("--model_path", type=str, default="iic/CosyVoice2-0.5B", help="HuggingFace model path for CosyVoice.")
    parser.add_argument("--commands_file", type=str, required=True, help="Path to a txt file with commands (one per line).")
    parser.add_argument("--instances_per_gpu", type=int, default=3, help="Number of model instances per GPU.")
    
    args = parser.parse_args()

    # Load commands
    if not os.path.exists(args.commands_file):
        print(f"Error: Commands file not found at {args.commands_file}")
        return

    with open(args.commands_file, 'r', encoding='utf-8') as f:
        # Filter out empty lines and comments
        command_lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    
    print(f"Loaded {len(command_lines)} commands from {args.commands_file}")
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    for command in command_lines:
        safe_command = safe_filename(command)
        os.makedirs(os.path.join(args.output_dir, safe_command), exist_ok=True)

    # Scan for wav files
    sample_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith('.wav')]
    print(f"Found {len(sample_files)} source samples in {args.input_dir}")

    if not sample_files:
        print("No .wav files found in input directory.")
        return

    # Resource Configuration
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_workers = num_gpus if num_gpus > 0 else 1 
    
    print(f"Starting execution with {num_workers} workers (GPUs), {args.instances_per_gpu} instances per worker.")

    # Queue Setup
    manager = multiprocessing.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    # Enqueue tasks
    for sample_file in sample_files:
        task_queue.put((sample_file, args.output_dir, command_lines, args.input_dir))

    for _ in range(num_workers):
        task_queue.put(None)

    # Start Processes
    processes = []
    for i in range(num_workers):
        gpu_id = i if num_gpus > 0 else -1
        p = multiprocessing.Process(
            target=worker_process,
            args=(gpu_id, args.instances_per_gpu, args.model_path, task_queue, result_queue))
        p.daemon = True
        p.start()
        processes.append(p)

    # Progress Bar
    success_count = 0
    failed_count = 0
    total_generated = 0
    completed_workers = 0
    
    with tqdm(total=len(sample_files), desc="Processing Samples") as pbar:
        while completed_workers < num_workers:
            result = result_queue.get()
            if result is None:
                completed_workers += 1
                continue
            
            success, sample_name, info = result
            if success:
                success_count += 1
                total_generated += info
            else:
                failed_count += 1
                
            pbar.update(1)
            pbar.set_postfix({"Gen": total_generated, "Fail": failed_count})

    for p in processes:
        p.join()

    print(f"\nDone! Success: {success_count}, Failed: {failed_count}, Total Files: {total_generated}")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()