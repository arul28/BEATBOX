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
import json
from tqdm import tqdm
import pickle

class AudioDataset(Dataset):
    def __init__(self, segment_info_path, transform=None, train=True, stats_file=None):
        self.df = pd.read_csv(segment_info_path)
        self.transform = transform
        self.train = train
        
        # Standardize JSo/JoS naming
        self.df['participant_id'] = self.df['participant_id'].replace('JoSP', 'JSoP')
        
        self.label_to_id = {'hhc': 0, 'hho': 1, 'kd': 2, 'sd': 3}
        
        # Parameters for mel spectrogram (as per paper)
        self.sample_rate = 16000
        self.n_mels = 64  # Changed from 128 to 64 to match paper
        self.n_fft = 1024
        self.hop_length = 512
        
        # Parameters for augmentation (10x as per paper)
        self.pitch_shifts = [-2, -1, 0, 1, 2]  # 5 pitch shifts
        self.time_stretches = [0.9, 0.95, 1.0, 1.05, 1.1]  # 5 time stretches
        
        # For global normalization
        if stats_file and os.path.exists(stats_file):
            # Load precomputed statistics from file
            with open(stats_file, 'rb') as f:
                stats = pickle.load(f)
                self.global_mean = stats['mean']
                self.global_std = stats['std']
            print(f"Loaded global statistics from {stats_file}")
            print(f"Global mean: {self.global_mean:.4f}, Global std: {self.global_std:.4f}")
        else:
            # Initialize with None, will be computed if needed
            self.global_mean = None
            self.global_std = None
    
    def compute_global_stats(self, save_path=None):
        """
        Compute global mean and std across all training spectrograms
        This is a one-time operation that should be run before training
        """
        print("Computing global spectrogram statistics...")
        all_values = []
        
        # Sample a subset of the data to speed up computation if dataset is large
        max_samples = min(1000, len(self.df))
        indices = np.random.choice(len(self.df), max_samples, replace=False)
        
        for i in tqdm(indices):
            row = self.df.iloc[i]
            audio_path = row['segment_path']
            
            # Generate log mel spectrogram without normalization
            mel_spec = self._compute_log_mel_spec(audio_path)
            all_values.extend(mel_spec.flatten().tolist())
            
        # Compute global statistics
        self.global_mean = np.mean(all_values)
        self.global_std = np.std(all_values)
        
        print(f"Global mean: {self.global_mean:.4f}, Global std: {self.global_std:.4f}")
        
        # Save statistics to file if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({'mean': self.global_mean, 'std': self.global_std}, f)
            print(f"Saved global statistics to {save_path}")
        
        return self.global_mean, self.global_std
    
    def _compute_log_mel_spec(self, audio_path, pitch_shift=0, time_stretch=1.0):
        """Compute log mel spectrogram without normalization"""
        # Fix path to use absolute path
        segments_base_path = 'C:/Users/arul2/OneDrive/Documents/Programming/BEATBOX/projectFiles/segments/'
        file_name = os.path.basename(audio_path)
        full_audio_path = os.path.join(segments_base_path, file_name)
        
        # Load audio
        try:
            y, sr = librosa.load(full_audio_path, sr=self.sample_rate)
        except Exception as e:
            print(f"Error loading {full_audio_path}: {str(e)}")
            raise
            
        # Apply augmentation
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
        
        return log_mel_spec
    
    def __len__(self):
        if self.train:
            return len(self.df) * len(self.pitch_shifts) * len(self.time_stretches)
        return len(self.df)
    
    def _load_and_process_audio(self, audio_path, pitch_shift=0, time_stretch=1.0):
        """Load and process audio with global standardization"""
        # Compute log mel spectrogram without normalization
        log_mel_spec = self._compute_log_mel_spec(audio_path, pitch_shift, time_stretch)
        
        # Apply global standardization (x - μ) / σ instead of per-patch min-max
        if self.global_mean is not None and self.global_std is not None:
            log_mel_spec = (log_mel_spec - self.global_mean) / self.global_std
        else:
            print("Warning: Global statistics not computed. Using raw log-mel values.")
        
        # Convert to tensor and ensure fixed size (64 x 48)
        spec_tensor = torch.FloatTensor(log_mel_spec)
        target_length = 48  # Changed from 64 to 48 to match paper
        current_length = spec_tensor.size(1)
        
        if current_length < target_length:
            pad_amount = target_length - current_length
            spec_tensor = torch.nn.functional.pad(spec_tensor, (0, pad_amount))
        elif current_length > target_length:
            start = (current_length - target_length) // 2
            spec_tensor = spec_tensor[:, start:start + target_length]
        
        return spec_tensor
    
    def __getitem__(self, idx):
        if self.train:
            n_augs = len(self.pitch_shifts) * len(self.time_stretches)
            orig_idx = idx // n_augs
            aug_idx = idx % n_augs
            pitch_idx = aug_idx // len(self.time_stretches)
            time_idx = aug_idx % len(self.time_stretches)
            
            pitch_shift = self.pitch_shifts[pitch_idx]
            time_stretch = self.time_stretches[time_idx]
        else:
            orig_idx = idx
            pitch_shift = 0
            time_stretch = 1.0
            
        row = self.df.iloc[orig_idx]
        audio_path = row['segment_path']
        label = self.label_to_id[row['instrument_label']]
        participant = row['participant_id']
        
        spec = self._load_and_process_audio(audio_path, pitch_shift, time_stretch)
        return spec, label, participant

class DrumCNN(nn.Module):
    def __init__(self, num_classes=4, embedding_dim=128):
        super(DrumCNN, self).__init__()
        
        # Modified architecture to match paper:
        # - Filter counts: 8→16→32→64 (instead of 64→128→256→512)
        # - Input shape: 64x48 mel spectrogram
        # - Embedding dimension: 128 (reduced from 256)
        # - Dropout: 0.3 (increased from 0.2)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8 x 32 x 24
            
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 x 16 x 12
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 x 8 x 6
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 64 x 4 x 3
        )
        
        self.flatten_size = 64 * 4 * 3
        
        self.embedding = nn.Sequential(
            nn.Linear(self.flatten_size, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x, return_embedding=False):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        embedding = self.embedding(x)
        
        if return_embedding:
            return embedding
            
        return self.classifier(embedding)

def get_device():
    """Check for GPU availability including 4090"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            if "4090" in gpu_name:
                print(f"Found NVIDIA RTX 4090: {gpu_name}")
                return torch.device(f"cuda:{i}")
        print(f"Using available GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda:0")
    elif platform.system() == "Darwin":  # macOS
        if torch.backends.mps.is_available():
            print("Using Apple Silicon GPU")
            return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")

def train_model(train_loader, val_loader, model, device, num_epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    
    best_val_acc = 0
    patience = 10  # Early stopping as per paper
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for specs, labels, _ in train_loader:
            specs = specs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for specs, labels, _ in val_loader:
                specs = specs.to(device)
                labels = labels.to(device)
                
                outputs = model(specs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.3f} | Train Acc: {train_acc:.3f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.3f} | Val Acc: {val_acc:.3f}%')
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_global_norm.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Verify participant splits
    print("\nVerifying dataset split:")
    train_participants = set()
    val_participants = set()
    for _, _, parts in train_loader:
        train_participants.update(parts)
        break
    for _, _, parts in val_loader:
        val_participants.update(parts)
        break
    print(f"Train participants: {sorted(list(train_participants))}")
    print(f"Val participants: {sorted(list(val_participants))}")
    print(f"Participant overlap: {train_participants.intersection(val_participants)}")

def analyze_participants(df):
    """Detailed analysis of participants in both datasets"""
    # Get all unique participant IDs directly from the CSV
    all_participants = sorted(df['participant_id'].unique())
    
    # Split into AVP and LVT participants
    avp_participants = sorted([p for p in all_participants if p.startswith('P')])
    lvt_participants = sorted([p for p in all_participants if not p.startswith('P')])
    
    print("\nDetailed Participant Analysis:")
    print(f"AVP Dataset ({len(avp_participants)} participants):")
    print(avp_participants)
    
    print(f"\nLVT Dataset ({len(lvt_participants)} participants):")
    print(lvt_participants)
    
    total_participants = len(all_participants)
    print(f"\nTotal unique participants: {total_participants}")
    print(f"- AVP Dataset: {len(avp_participants)} participants")
    print(f"- LVT Dataset: {len(lvt_participants)} participants")
    
    return avp_participants, lvt_participants

def main():
    # Set device and random seed
    device = get_device()
    torch.manual_seed(42)
    
    # Paths
    segment_info_path = 'C:/Users/arul2/OneDrive/Documents/Programming/BEATBOX/projectFiles/segment_info/segment_info_base_names.csv'
    stats_file = 'C:/Users/arul2/OneDrive/Documents/Programming/BEATBOX/projectFiles/code/models/cnn/global_mel_stats.pkl'
    
    # Create dataset
    dataset = AudioDataset(
        segment_info_path=segment_info_path,
        train=True,
        stats_file=stats_file if os.path.exists(stats_file) else None
    )
    
    # Compute global statistics if they don't exist
    if not os.path.exists(stats_file):
        dataset.compute_global_stats(save_path=stats_file)
    
    # Analyze participants
    avp_participants, lvt_participants = analyze_participants(dataset.df)
    
    # Create participant groups for splitting
    # Use participants directly from the CSV file without special handling
    groups = dataset.df['participant_id'].values
    
    # Cross-validation setup
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    
    # Storage for embeddings and labels
    all_embeddings = []
    all_labels = []
    all_participants = []
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X=np.zeros(len(dataset.df)), groups=groups)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)
        
        model = DrumCNN().to(device)
        train_model(train_loader, val_loader, model, device)
        
        # Extract embeddings
        model.eval()
        fold_embeddings = []
        fold_labels = []
        fold_participants = []
        
        with torch.no_grad():
            for specs, labels, parts in val_loader:
                specs = specs.to(device)
                embeddings = model(specs, return_embedding=True)
                
                fold_embeddings.append(embeddings.cpu().numpy())
                fold_labels.extend(labels.numpy())  # Changed from append to extend
                fold_participants.extend(parts)
        
        all_embeddings.append(np.concatenate(fold_embeddings))
        all_labels.extend(fold_labels)  # Changed from append to extend
        all_participants.extend(fold_participants)
    
    # Prepare for k-NN evaluation
    embeddings = np.concatenate(all_embeddings)
    labels = np.array(all_labels)  # Now this should work
    participants = np.array(all_participants)
    
    print("\nFinal dataset sizes:")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Participants shape: {participants.shape}")
    
    # Scale embeddings (as per paper)
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    
    # Save embeddings, labels and participants to files
    output_dir = 'C:/Users/arul2/OneDrive/Documents/Programming/BEATBOX/projectFiles/code/models/cnn'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the raw data with global_norm suffix
    np.save(os.path.join(output_dir, 'embeddings_global_norm.npy'), embeddings)
    np.save(os.path.join(output_dir, 'instrument_labels_global_norm.npy'), labels)
    np.save(os.path.join(output_dir, 'participants_global_norm.npy'), participants)
    
    # Also save the scaled embeddings
    np.save(os.path.join(output_dir, 'scaled_embeddings_global_norm.npy'), scaled_embeddings)
    
    # Save speaker splits for reference
    with open(os.path.join(output_dir, 'splits_global_norm.json'), 'w') as f:
        json.dump({
            'avp_participants': avp_participants,
            'lvt_participants': lvt_participants,
            'all_participants': sorted(avp_participants + lvt_participants)
        }, f, indent=2)
    
    print(f"\nSaved embeddings and metadata to {output_dir}:")
    print(f"- embeddings_global_norm.npy: {embeddings.shape}")
    print(f"- scaled_embeddings_global_norm.npy: {scaled_embeddings.shape}")
    print(f"- instrument_labels_global_norm.npy: {labels.shape}")
    print(f"- participants_global_norm.npy: {participants.shape}")
    
    # Evaluate using leave-one-participant-out
    knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
    unique_participants = np.unique(participants)
    participant_accuracies = []
    
    print("\nParticipant-wise evaluation:")
    for test_participant in unique_participants:
        train_mask = participants != test_participant
        test_mask = participants == test_participant
        
        # Scale using only training data
        train_embeddings = scaled_embeddings[train_mask]
        test_embeddings = scaled_embeddings[test_mask]
        
        knn.fit(train_embeddings, labels[train_mask])
        accuracy = knn.score(test_embeddings, labels[test_mask])
        participant_accuracies.append(accuracy)
        
        print(f"Participant {test_participant}: {accuracy:.3f}")
    
    print(f"\nMean participant-independent accuracy: {np.mean(participant_accuracies):.3f}")

if __name__ == "__main__":
    main() 