"""Load precomputed MOSI/MOSEI features in the MMSA pickle format."""
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class SplitDataset(Dataset):
    def __init__(self, split):
        self.text = split['text_bert'].astype(np.float32)
        self.audio = split['audio'].astype(np.float32)
        self.vision = split['vision'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio_lengths = split['audio_lengths']
        self.vision_lengths = split['vision_lengths']
        self.labels = split['regression_labels'].astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return dict(text=torch.from_numpy(self.text[index]),
                    audio=torch.from_numpy(self.audio[index]),
                    vision=torch.from_numpy(self.vision[index]),
                    audio_lengths=torch.as_tensor(self.audio_lengths[index]),
                    vision_lengths=torch.as_tensor(self.vision_lengths[index]),
                    labels={'M': torch.from_numpy(self.labels[index].reshape(-1))},
                    index=torch.as_tensor(index))

def load_splits(path):
    # Pickle input must come from a trusted dataset provider.
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return {split: SplitDataset(data[split]) for split in ('train', 'valid', 'test')}
