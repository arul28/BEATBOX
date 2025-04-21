import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
import os
from pathlib import Path
import platform
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import time
from datetime import datetime, timedelta
import warnings
import random
warnings.filterwarnings('ignore')  # Hide warnings

# Global constants
AUG_DIR = 'C:/Users/arul2/OneDrive/Documents/Programming/BEATBOX/projectFiles/code/models/cnn/aug_dir'  # Directory for augmented spectrograms

class AudioDataset(Dataset):
    def __init__(self, segment_info_path, transform=None, train=True):
        self.df = pd.read_csv(segment_info_path)
        self.transform = transform
        self.train = train
        
        # Create mapping for instrument labels
        unique_instruments = sorted(self.df['instrument_label'].unique())
        self.instrument_to_id = {inst: idx for idx, inst in enumerate(unique_instruments)}
        self.num_classes = len(unique_instruments)
        
        # Debug: Print label mapping and counts
        print("\n=== Label Mapping Debug ===")
        print("→ label mapping:", self.instrument_to_id)
        print("  label counts:", self.df['instrument_label'].value_counts().to_dict())
        assert set(self.instrument_to_id.values()) == set(range(self.num_classes)), \
            f"Labels should be 0 to {self.num_classes-1}, got {self.instrument_to_id.values()}"
        print("✓ Label mapping verified")
        
        # Fix spectrogram parameters
        self.sample_rate = 44100  # 44.1 kHz
        self.n_mels = 64     # 64 mel bins
        self.n_fft = 2048    # For 46ms window at 44.1kHz
        self.hop_length = 512  # Exactly 12ms hop at 44.1kHz
        self.patch_duration = 0.56  # seconds
        
        # Standardize JSo/JoS naming
        self.df['participant_id'] = self.df['participant_id'].replace('JoSP', 'JSoP')
        
        # Create efficient path lookup
        self.path2row = {}
        for i, path in enumerate(self.df['segment_path']):
            self.path2row[os.path.basename(path)] = i
        
        # OVERFITTING TEST: Use only one speaker with 50 examples
        small_df = self.df[self.df.participant_id == 'P1'].sample(50, random_state=42)
        self.df = small_df
        print(f"\n=== Overfitting Test Setup ===")
        print(f"Using {len(self.df)} examples from speaker P1")
        print("Label distribution:", self.df['instrument_label'].value_counts().to_dict())
        
        # OVERFITTING TEST: Disable augmentation
        self.augmentation_params = [{'pitch_shift': 0.0, 'time_stretch': 1.0}]  # Only original
        
        # Fix the segments path to use absolute path with forward slashes
        self.segments_base_path = 'C:/Users/arul2/OneDrive/Documents/Programming/BEATBOX/projectFiles/segments/'
        
        self.print_counter = 0  # Add counter for limiting prints
        self.print_frequency = 100  # Only print every 100th file
        
    def __len__(self):
        if self.train:
            return len(self.df) * len(self.augmentation_params)  # Each file gets 11 versions
        return len(self.df)
    
    def _load_and_process_audio(self, audio_path, pitch_shift=0, time_stretch=1.0):
        """Load and process audio following paper's normalization"""
        # Get file basename for lookup
        filename = os.path.basename(audio_path) if "segments/" in audio_path else os.path.basename(audio_path)
        
        # Find the row in the filtered DataFrame
        row = self.df[self.df['segment_path'].str.endswith(filename)].iloc[0]
        
        # Get onset time from row and convert to sample
        # Handle both seconds and milliseconds formats
        onset_time = float(row['onset_time'])
        
        # Check if the time is likely in milliseconds (heuristic: if it's a large value)
        if onset_time > 1000:  # Assume this is milliseconds
            onset_time = onset_time / 1000.0  # Convert to seconds
        
        onset_sample = int(onset_time * self.sample_rate)
        
        # Add a sanity check for onset_sample
        if onset_sample < 0:
            onset_sample = 0
            print(f"Warning: Negative onset time detected in {filename}. Using start of file.")
        
        # Create full path to audio file
        if "segments/" in audio_path:
            audio_path = os.path.join(self.segments_base_path, filename)
        else:
            audio_path = os.path.join(self.segments_base_path, audio_path)
            
        # Only print occasionally
        self.print_counter += 1
        if self.print_counter % self.print_frequency == 0:
            print(f"Processing audio files... (processed {self.print_counter} files)")
        
        try:
            # Try soundfile first
            import soundfile as sf
            y, sr = sf.read(audio_path)
            if sr != self.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
        except:
            try:
                # Try torchaudio as fallback
                waveform, sr = torchaudio.load(audio_path)
                y = waveform[0].numpy()
                if sr != self.sample_rate:
                    y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            except:
                # Use audioread as last resort
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Calculate patch length in samples
        patch_len = int(self.patch_duration * self.sample_rate)
        
        # First slice the audio at the onset
        onset_sample = min(max(0, onset_sample), len(y) - 1)
        end_sample = min(onset_sample + patch_len, len(y))
        y = y[onset_sample:end_sample]
        
        # Add zero-padding if the segment is too short
        if len(y) < patch_len:
            padding = patch_len - len(y)
            y = np.pad(y, (0, padding), 'constant')
        
        # Now apply augmentations to the sliced segment
        if time_stretch != 1.0:
            y = librosa.effects.time_stretch(y, rate=time_stretch)
        if pitch_shift != 0:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
            
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec)
        
        # Verify mel spectrogram shape
        if log_mel_spec.shape[0] != self.n_mels:
            raise ValueError(f"Expected {self.n_mels} mel bins, got {log_mel_spec.shape[0]}")
        
        # Convert to tensor and ensure shape
        spec_tensor = torch.FloatTensor(log_mel_spec)  # Shape: [n_mels, frames]
        
        # Ensure exactly 48 frames
        if spec_tensor.size(1) < 48:
            pad = 48 - spec_tensor.size(1)
            spec_tensor = torch.nn.functional.pad(spec_tensor, (0, pad, 0, 0))
        elif spec_tensor.size(1) > 48:
            spec_tensor = spec_tensor[:, :48]
        
        # Final shape verification
        if spec_tensor.shape != (64, 48):
            raise ValueError(f"Invalid spectrogram shape: {spec_tensor.shape}, expected (64, 48)")
        
        return spec_tensor
    
    def __getitem__(self, idx):
        if self.train:
            n_augs = len(self.augmentation_params)
            orig_idx = idx // n_augs  # Calculate the original index first
            aug_idx = idx % n_augs
            
            # Verify orig_idx is within bounds
            if orig_idx >= len(self.df):
                raise IndexError(f"Index {idx} out of bounds (orig_idx {orig_idx} >= {len(self.df)})")
            
            params = self.augmentation_params[aug_idx]
            pitch_shift = params['pitch_shift']
            time_stretch = params['time_stretch']
        else:
            orig_idx = idx
            pitch_shift = 0
            time_stretch = 1.0
            
        row = self.df.iloc[orig_idx]  # Use orig_idx here
        audio_path = row['segment_path']
        label = self.instrument_to_id[row['instrument_label']]
        participant = row['participant_id']
        
        spec = self._load_and_process_audio(audio_path, pitch_shift, time_stretch)
        return spec, label, participant

class AugmentedDataset(Dataset):
    def __init__(self, aug_dir, speakers, augmented=True):
        """
        Dataset for loading spectrograms
        
        Parameters:
        -----------
        aug_dir: str
            Directory containing spectrograms
        speakers: list
            List of participant IDs to include
        augmented: bool
            If True, include augmented files
            If False, include only original (non-augmented) files
        """
        self.aug_dir = aug_dir
        self.speakers = set(speakers)
        self.augmented = augmented
        
        # Load the base dataset to get the number of classes
        base_dataset = AudioDataset(
            segment_info_path='C:/Users/arul2/OneDrive/Documents/Programming/BEATBOX/projectFiles/segment_info/segment_info_base_names.csv',
            train=True
        )
        self.num_classes = base_dataset.num_classes
        print(f"Number of instrument classes: {self.num_classes}")
        
        # Check if augmentation directory exists and is empty
        if not os.path.exists(aug_dir) or not os.listdir(aug_dir):
            print("Augmentation directory is empty or missing. Regenerating augmentations...")
            # Generate augmentations
            generate_augmentations(base_dataset)
            print("Augmentations regenerated successfully!")
        
        # List all files for these speakers
        self.files = []
        
        for file in os.listdir(aug_dir):
            try:
                data = torch.load(os.path.join(aug_dir, file))
                
                # Filter by participant
                if data['participant'] in self.speakers:
                    # Filter by augmentation status
                    if self.augmented:
                        # Include both original and augmented files for training
                        self.files.append(file)
                    else:
                        # Include only original files (without _aug) for validation
                        if '_aug' not in file:
                            self.files.append(file)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        # Validate first file to catch issues early
        if self.files:
            try:
                test_data = torch.load(os.path.join(aug_dir, self.files[0]))
                if not isinstance(test_data['spectrogram'], torch.Tensor):
                    raise ValueError("Spectrogram is not a tensor")
                if test_data['spectrogram'].shape != (64, 48):
                    raise ValueError(f"Invalid spectrogram shape: {test_data['spectrogram'].shape}")
                if not isinstance(test_data['label'], torch.Tensor):
                    raise ValueError("Label is not a tensor")
                # Validate label range
                label_val = test_data['label'].item() if isinstance(test_data['label'], torch.Tensor) else test_data['label']
                if not (0 <= label_val < self.num_classes):
                    raise ValueError(f"Invalid label value: {label_val}, should be in range [0, {self.num_classes})")
            except Exception as e:
                raise ValueError(f"Dataset validation failed: {str(e)}")
        else:
            raise ValueError("No valid files found in augmentation directory")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.aug_dir, self.files[idx]))
        spec = data['spectrogram']
        
        # Ensure label is a tensor and within valid range
        if isinstance(data['label'], torch.Tensor):
            label = data['label']
        else:
            label = torch.tensor(data['label'], dtype=torch.long)
        
        # Verify label is within valid range
        label_val = label.item() if isinstance(label, torch.Tensor) else label
        if not (0 <= label_val < self.num_classes):
            raise ValueError(f"Invalid label value at index {idx}: {label_val}, should be in range [0, {self.num_classes})")
        
        # Verify shapes
        if spec.shape != (64, 48):
            raise ValueError(f"Invalid spectrogram shape at index {idx}: {spec.shape}")
        
        return spec, label, data['participant']

class DrumCNN(nn.Module):
    def __init__(self, num_classes, embedding_dim=1024):
        super(DrumCNN, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                # First conv in block
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
            nn.ReLU(),
                # Second conv in block
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv_layers = nn.Sequential(
            conv_block(1, 8),    # First block: 1 → 8
            conv_block(8, 16),   # Second block: 8 → 16
            conv_block(16, 32),  # Third block: 16 → 32
            conv_block(32, 64)   # Fourth block: 32 → 64
        )
        
        # Correct flatten size: 64 channels × 4 × 3 after four 2×2 pools
        self.flatten_size = 64 * 4 * 3  # 768 units
        
        self.embedding = nn.Sequential(
            nn.Linear(self.flatten_size, embedding_dim),
            nn.ReLU()
            # No dropout as per paper
        )
        
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x, return_embedding=False):
        # Input x has shape [batch, 64, 48] - mel spectrograms
        # Need to transpose from [batch, freq, time] to [batch, 1, freq, time]
        x = x.unsqueeze(1)  # Add channel dimension: [batch, 1, 64, 48]
        
        # Apply convolution layers
        x = self.conv_layers(x)  # Output: [batch, 64, 4, 3]
        
        # Flatten for FC layer
        x = x.view(x.size(0), -1)  # [batch, 64 * 4 * 3] = [batch, 768]
        
        # Get embedding
        embedding = self.embedding(x)  # [batch, 1024]
        
        if return_embedding:
            return embedding
            
        # Get class predictions
        return self.classifier(embedding)  # [batch, num_classes]

def get_device():
    """Check for GPU availability including 4090"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            if "4090" in gpu_name:
                print(f"Found NVIDIA RTX 4090: {gpu_name}")
                return torch.device(f"cuda:{i}"), True
        print(f"Using available GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda:0"), True
    elif platform.system() == "Darwin":  # macOS
        if torch.backends.mps.is_available():
            print("Using Apple Silicon GPU")
            return torch.device("mps"), False
    print("Using CPU")
    return torch.device("cpu"), False

def get_gpu_memory_usage():
    """Get GPU memory usage in GB if available"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        return f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved"
    return "GPU not available"

def train_model(train_loader, val_loader, model, device, cuda_available=False, num_epochs=100, output_dir=None):
    # Validate first batch before starting training
    print("Validating first batch...")
    try:
        test_batch = next(iter(train_loader))
        specs, labels, _ = test_batch
        print(f"Batch shapes - Specs: {specs.shape}, Labels: {labels.shape}")
        
        # Debug: Print label information for first batch
        print("\n=== First Batch Label Debug ===")
        print("→ Unique labels in batch:", torch.unique(labels).tolist())
        print("→ Label counts in batch:", torch.bincount(labels).tolist())
        print("→ Label range check:", f"min={labels.min().item()}, max={labels.max().item()}")
        
        if not isinstance(specs, torch.Tensor):
            raise ValueError("Specs is not a tensor")
        if not isinstance(labels, torch.Tensor):
            raise ValueError("Labels is not a tensor")
            
        # Check for batch dimension and correct feature dimensions
        if len(specs.shape) != 3:  # Should be [batch_size, 64, 48]
            raise ValueError(f"Expected 3 dimensions [batch_size, 64, 48], got shape: {specs.shape}")
        if specs.shape[1:] != (64, 48):
            raise ValueError(f"Expected feature dimensions [64, 48], got: {specs.shape[1:]}")
            
    except Exception as e:
        raise ValueError(f"Batch validation failed: {str(e)}")
    
    print("Batch validation successful!")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Increased learning rate for overfitting
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    scaler = GradScaler('cuda') if cuda_available else None
    
    print("\nStarting training loop with detailed debugging...")
    print(f"Number of batches: {len(train_loader)}")
    print(f"Device: {device}")
    print(f"CUDA available: {cuda_available}")
    print(f"Learning rate: 1e-3")
    
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\n{'-'*20}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        train_pbar = tqdm(
            train_loader, 
            desc=f"Training",
            ncols=100,  # Fixed width
            leave=True  # Keep the progress bar
        )
        
        for batch_idx, (specs, labels, _) in enumerate(train_pbar):
            try:
                specs = specs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                if cuda_available:
                    with autocast(device_type='cuda'):
                        outputs = model(specs)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(specs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar with concise information
                train_acc = 100. * correct / total
                train_pbar.set_postfix({
                    'loss': f"{train_loss/total:.3f}",
                    'acc': f"{train_acc:.2f}%",
                    'gpu_mem': f"{torch.cuda.memory_allocated()/1e9:.1f}GB" if cuda_available else "N/A"
                }, refresh=True)
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {str(e)}")
                raise e
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss/len(train_loader):.3f} | Train Acc: {train_acc:.2f}%")
        
        # Validation phase with similar concise output
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        val_pbar = tqdm(val_loader, desc="Validation", ncols=100, leave=True)
        with torch.no_grad():
            for specs, labels, _ in val_pbar:
                specs = specs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                if cuda_available:
                    with autocast(device_type='cuda'):
                        outputs = model(specs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(specs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                val_acc = 100. * correct / total
                val_pbar.set_postfix({
                    'loss': f"{val_loss/total:.3f}",
                    'acc': f"{val_acc:.2f}%"
                }, refresh=True)
        
        val_acc = 100. * correct / total
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss/len(train_loader):.3f} | Train Acc: {train_acc:.3f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.3f} | Val Acc: {val_acc:.3f}%")
        
        if cuda_available:
            print(get_gpu_memory_usage())
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            patience_counter = 0
            model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"✅ New best model saved to {model_path}! Validation accuracy improved by {improvement:.2f}%")
        else:
            patience_counter += 1
            remaining_patience = patience - patience_counter
            print(f"⚠️ Validation accuracy did not improve. Patience: {remaining_patience}/{patience}")
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break
    
    return best_val_acc

def generate_augmentations(dataset):
    """Generate original and 10 augmented versions of each clip"""
    print("\n=== Starting Augmentation Process ===")
    
    # Validate dataset size
    df_len = len(dataset.df)
    aug_len = len(dataset.augmentation_params)
    total_len = len(dataset)
    print(f"\nDataset validation:")
    print(f"Number of original files: {df_len}")
    print(f"Number of augmentations per file: {aug_len}")
    print(f"Total dataset length: {total_len}")
    
    if total_len != df_len * aug_len:
        raise ValueError(f"Dataset length mismatch: {total_len} != {df_len} * {aug_len}")
    
    # Check if augmentation directory exists and has valid files
    if os.path.exists(AUG_DIR):
        existing_files = os.listdir(AUG_DIR)
        if existing_files:
            # Check if we have the expected number of files
            expected_total = df_len * len(dataset.augmentation_params)
            if len(existing_files) == expected_total:
                print(f"Found existing augmentations with correct number of files ({len(existing_files)})")
                print("Skipping augmentation generation...")
                return
            
            print(f"Found {len(existing_files)} files, expected {expected_total}")
            print("Regenerating augmentations...")
            # Clear existing augmentation directory
            for file in existing_files:
                os.remove(os.path.join(AUG_DIR, file))
    
    # Create directory for augmented spectrograms
    os.makedirs(AUG_DIR, exist_ok=True)
    
    # Use df_len instead of len(dataset) to iterate only over original files
    total_files = df_len
    expected_total = total_files * len(dataset.augmentation_params)
    print(f"\nProcessing {total_files} original files")
    print(f"Generating {len(dataset.augmentation_params)} versions per file (1 original + {len(dataset.augmentation_params)-1} augmented)")
    print(f"Expected total files: {expected_total}")
    
    generated_count = 0
    error_count = 0
    
    for idx in tqdm(range(total_files), desc="Generating augmentations"):
        try:
            # Get original file data
            row = dataset.df.iloc[idx]
            audio_path = row['segment_path']
            label = dataset.instrument_to_id[row['instrument_label']]
            participant = row['participant_id']
            
            # Convert label to tensor if it isn't already
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, dtype=torch.long)
            
            base_name = os.path.basename(audio_path)
            
            # Process each augmentation parameter
            for aug_idx, params in enumerate(dataset.augmentation_params):
                try:
                    aug_spec = dataset._load_and_process_audio(
                        audio_path,
                        pitch_shift=params['pitch_shift'],
                        time_stretch=params['time_stretch']
                    )
                    
                    # Save with appropriate suffix (no suffix for original)
                    if aug_idx == 0:
                        save_path = f"{AUG_DIR}/{base_name.replace('.wav', '.pt')}"
                    else:
                        save_path = f"{AUG_DIR}/{base_name.replace('.wav', f'_aug{aug_idx}.pt')}"
                    
                    torch.save({
                        'spectrogram': aug_spec,
                        'label': label,
                        'participant': participant
                    }, save_path)
                    generated_count += 1
                    
                except Exception as e:
                    error_count += 1
                    print(f"\nError generating augmentation {aug_idx} for file {base_name}: {str(e)}")
                    continue
                    
        except Exception as e:
            error_count += 1
            print(f"\nError processing file at index {idx}: {str(e)}")
            continue
    
    # Verify results
    actual_files = len(os.listdir(AUG_DIR))
    print("\n=== Augmentation Complete ===")
    print(f"Expected files: {expected_total}")
    print(f"Actually generated: {generated_count}")
    print(f"Files in directory: {actual_files}")
    print(f"Errors encountered: {error_count}")
    
    if actual_files != expected_total:
        print("\nWARNING: Number of generated files doesn't match expected count!")
        print("This might indicate issues in the augmentation process.")
    else:
        print("\nSuccess! Generated exactly the expected number of files.")

def create_speaker_splits(df, seed=42):
    """Create fixed splits for 40 training and 8 eval speakers (4 AVP, 4 LVT)"""
    # Get AVP participant IDs (those starting with 'P')
    avp_ids = sorted(df[df.participant_id.str.startswith('P')]['participant_id'].unique())
    
    # Get LVT participant IDs (those not starting with 'P')
    lvt_ids = sorted(df[~df.participant_id.str.startswith('P')]['participant_id'].unique())
    
    print(f"AVP IDs: {avp_ids}")
    print(f"LVT IDs: {lvt_ids}")
    print(f"Total unique participants: {len(avp_ids) + len(lvt_ids)}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Select 4 AVP participants for evaluation
    eval_avp = random.sample(avp_ids, 4)
    
    # Select 4 LVT participants for evaluation
    eval_lvt = random.sample(lvt_ids, 4)
    
    eval_speakers = eval_avp + eval_lvt
    
    # Get training speakers (all remaining participants)
    train_speakers = []
    
    # Add all AVP speakers except those in eval_avp
    for p in avp_ids:
        if p not in eval_avp:
            train_speakers.append(p)
    
    # Add all LVT speakers except those in eval_lvt
    for p in lvt_ids:
        if p not in eval_lvt:
            train_speakers.append(p)
    
    print(f"\nFINAL HOLD-OUT (8 speakers):")
    print(f"AVP: {eval_avp}")
    print(f"LVT: {eval_lvt}")
    print(f"\nTRAINING SET ({len(train_speakers)} speakers):")
    print(f"AVP: {[p for p in train_speakers if p.startswith('P')]}")
    print(f"LVT: {[p for p in train_speakers if not p.startswith('P')]}")
    
    return train_speakers, eval_speakers

def create_speaker_model_mapping(train_speaker_folds):
    """Create mapping from speaker to model index"""
    speaker_to_model = {}
    for fold_idx, fold_split in enumerate(train_speaker_folds):
        for speaker in fold_split['val']:
            speaker_to_model[speaker] = fold_idx
    return speaker_to_model

def extract_embeddings(dataset, models, speaker_to_model, device, cuda_available, eval_speakers=None):
    """Extract embeddings using appropriate model for each speaker
    
    For training speakers: use the model trained when they were in validation
    For evaluation speakers: average embeddings from all models (as per paper)
    """
    all_embeddings = []
    all_labels = []
    all_participants = []
    
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=cuda_available
    )
    
    for specs, labels, participants in tqdm(loader, desc="Extracting embeddings"):
        # Convert to numpy arrays for consistent handling
        specs_np = specs.numpy() if isinstance(specs, torch.Tensor) else np.array(specs)
        labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
        participants_np = np.array(participants)  # Convert list of strings to numpy array
        
        # Group by speaker
        unique_speakers = np.unique(participants_np)
        
        for speaker in unique_speakers:
            # Create mask for current speaker
            mask = participants_np == speaker
            
            # Get data for current speaker
            speaker_specs = torch.tensor(specs_np[mask]).to(device)
            speaker_labels = labels_np[mask]
            
            if speaker in speaker_to_model:
                # Training speaker: use appropriate model
                model_idx = speaker_to_model[speaker]
                model = models[model_idx]
                model.eval()
                
                with torch.no_grad():
                    if cuda_available:
                        with autocast(device_type='cuda'):
                            emb = model(speaker_specs, return_embedding=True)
                    else:
                        emb = model(speaker_specs, return_embedding=True)
                
                all_embeddings.append(emb.cpu().numpy())
                all_labels.extend(speaker_labels)
                all_participants.extend([speaker] * len(speaker_labels))
            else:
                # Eval speaker: average embeddings from all models
                speaker_embeddings = []
                
                for model in models:
                    model.eval()
                    with torch.no_grad():
                        if cuda_available:
                            with autocast(device_type='cuda'):
                                emb = model(speaker_specs, return_embedding=True)
                        else:
                            emb = model(speaker_specs, return_embedding=True)
                    speaker_embeddings.append(emb.cpu().numpy())
                
                # Average embeddings from all models
                avg_emb = np.mean(speaker_embeddings, axis=0)
                all_embeddings.append(avg_emb)
                all_labels.extend(speaker_labels)
                all_participants.extend([speaker] * len(speaker_labels))
    
    if all_embeddings:
        return np.concatenate(all_embeddings), np.array(all_labels), np.array(all_participants)
    else:
        return np.array([]), np.array([]), np.array([])

def validate_dataset(dataset):
    """Validate dataset before augmentation to catch issues early"""
    print("Validating dataset before augmentation...")
    try:
        # Check first item
        spec, label, participant = dataset[0]
        if not isinstance(spec, torch.Tensor):
            raise ValueError("Spectrogram is not a tensor")
        if spec.shape != (64, 48):
            raise ValueError(f"Invalid spectrogram shape: {spec.shape}")
        
        # Check a few random items
        for idx in np.random.choice(len(dataset), min(5, len(dataset)), replace=False):
            spec, label, participant = dataset[idx]
            if spec.shape != (64, 48):
                raise ValueError(f"Invalid spectrogram shape at index {idx}: {spec.shape}")
    except Exception as e:
        raise ValueError(f"Dataset validation failed: {str(e)}")
    
    print("Dataset validation successful!")
    return True

def verify_speaker_splits(train_speakers, val_speakers):
    """Verify that there is no overlap between train and validation speakers"""
    # Check for direct overlap (no need to get base names anymore)
    overlap = set(train_speakers).intersection(set(val_speakers))
    if overlap:
        raise ValueError(f"Found participant overlap between train and validation sets: {overlap}")
    
    print("\nSpeaker Split Verification:")
    print(f"Number of training speakers: {len(train_speakers)}")
    print(f"Number of validation speakers: {len(val_speakers)}")
    print(f"Training speakers: {sorted(train_speakers)}")
    print(f"Validation speakers: {sorted(val_speakers)}")
    print("✓ No participant overlap detected")
    return True

def main():
    # Create output directory
    output_dir = 'C:/Users/arul2/OneDrive/Documents/Programming/BEATBOX/projectFiles/code/models/cnn'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")
    
    print(f"\n{'='*20} BEATBOX DRUM RECOGNITION {'='*20}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    
    device, cuda_available = get_device()
    
    if cuda_available:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"CUDA optimizations enabled: {get_gpu_memory_usage()}")
    
    torch.manual_seed(42)
    print("Random seed set to 42")
    
    print(f"\n{'='*20} LOADING DATASET {'='*20}")
    print("Loading dataset from '../../../segment_info/segment_info_base_names.csv'...")
    
    # Load the full dataset first
    dataset = AudioDataset(
        segment_info_path='C:/Users/arul2/OneDrive/Documents/Programming/BEATBOX/projectFiles/segment_info/segment_info_base_names.csv',
        train=True  # Enable augmentation for full dataset
    )
    print(f"Dataset loaded with {len(dataset)} samples")
    print(f"Total instrument classes: {dataset.num_classes}")
    
    # Store the number of classes for consistent model initialization
    num_classes = dataset.num_classes
    
    # OVERFITTING TEST: Skip augmentation generation
    # print(f"\n{'='*20} GENERATING AUGMENTATIONS {'='*20}")
    # generate_augmentations(dataset)
    
    # OVERFITTING TEST: Use a simple train/val split
    from sklearn.model_selection import train_test_split
    
    # Get all examples from the dataset
    all_examples = list(range(len(dataset)))
    
    # Split into train and validation (80/20)
    train_indices, val_indices = train_test_split(all_examples, test_size=0.2, random_state=42)
    
    # Create data loaders
        train_loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=4,
            pin_memory=cuda_available,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices)
        )
        
        val_loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=4,
            pin_memory=cuda_available,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices)
    )
    
    # Train model
    model = DrumCNN(num_classes=num_classes).to(device)
    train_model(train_loader, val_loader, model, device, cuda_available, num_epochs=100, output_dir=output_dir)
    
    print("\nOverfitting test complete!")

if __name__ == "__main__":
    main()