import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import natsort

class FrameSequenceDataset(data.Dataset):
    def __init__(self, frames_root, gt_root=None, trainsize=416, sequence_length=4):
        """
        Dataset for loading frame sequences from directories
        Args:
            frames_root: Root directory containing frame sequence directories
            gt_root: Directory containing ground truth masks (optional)
            trainsize: Size to resize frames to
            sequence_length: Number of consecutive frames to include in each sequence
        """
        self.trainsize = trainsize
        self.sequence_length = sequence_length
        
        # Get all sequence directories
        self.sequence_dirs = [d for d in os.listdir(frames_root) 
                            if os.path.isdir(os.path.join(frames_root, d))]
        self.sequence_dirs = natsort.natsorted(self.sequence_dirs)  # Natural sort for proper ordering
        
        # Store full paths
        self.frames_root = frames_root
        self.gt_root = gt_root
        
        # Get frame paths for each sequence
        self.sequences = []
        for seq_dir in self.sequence_dirs:
            seq_path = os.path.join(frames_root, seq_dir)
            frames = [f for f in os.listdir(seq_path) if f.endswith(('.jpg', '.png'))]
            frames = natsort.natsorted(frames)  # Natural sort frame names
            
            # Create frame sequences
            for i in range(len(frames) - sequence_length + 1):
                frame_sequence = [os.path.join(seq_path, frames[j]) 
                                for j in range(i, i + sequence_length)]
                self.sequences.append({
                    'frames': frame_sequence,
                    'sequence_name': seq_dir,
                    'start_idx': i
                })
        
        # Image transformations
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # GT transformations
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Load frames
        frames = []
        for frame_path in sequence['frames']:
            frame = Image.open(frame_path).convert('RGB')
            frame = self.img_transform(frame)
            frames.append(frame)
        
        # Stack frames
        frames = torch.stack(frames, dim=0)
        
        if self.gt_root is not None:
            # Load corresponding ground truth if available
            # Assuming GT name matches the middle frame of the sequence
            mid_frame_idx = sequence['start_idx'] + self.sequence_length // 2
            gt_name = f"{sequence['sequence_name']}_{mid_frame_idx:04d}.png"
            gt_path = os.path.join(self.gt_root, gt_name)
            
            if os.path.exists(gt_path):
                gt = Image.open(gt_path).convert('L')
                gt = self.gt_transform(gt)
                return frames, gt
            else:
                print(f"Warning: GT not found for {gt_path}")
                gt = torch.zeros((1, self.trainsize, self.trainsize))
                return frames, gt
        
        return frames

def get_frame_sequence_loader(frames_root, gt_root=None, batch_size=1, trainsize=416, 
                            sequence_length=4, shuffle=True, num_workers=4):
    """
    Create a frame sequence data loader
    Args:
        frames_root: Root directory containing frame sequence directories
        gt_root: Directory containing ground truth masks (optional)
        batch_size: Number of sequences per batch
        trainsize: Size to resize frames to
        sequence_length: Number of consecutive frames to include in each sequence
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker threads for loading data
    """
    dataset = FrameSequenceDataset(frames_root, gt_root, trainsize, sequence_length)
    
    return data.DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers,
                          pin_memory=True)

# Example folder structure:
# frames_root/
#   sequence1/
#     frame_0001.jpg
#     frame_0002.jpg
#     ...
#   sequence2/
#     frame_0001.jpg
#     frame_0002.jpg
#     ...
# gt_root/
#   sequence1_0001.png
#   sequence1_0002.png
#   ...