"""
Reimplemented: Wooseok Shin
"""

import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from signal_processing import get_spec_and_phase

class Generator_train_dataset(Dataset):
    def __init__(self, file_list, noisy_path, num_target_metric=1):
        self.file_list = file_list
        self.noisy_path = noisy_path
        self.target_score = torch.ones(num_target_metric, dtype=torch.float32)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        clean_wav, _ = torchaudio.load(self.file_list[idx])
        noisy_wav, _ = torchaudio.load(self.noisy_path + os.path.basename(self.file_list[idx]))

        # [1, T, F]
        noisy_mag, noisy_phase = get_spec_and_phase(noisy_wav)
        clean_mag, clean_phase = get_spec_and_phase(clean_wav)

        return clean_mag, clean_phase, noisy_mag, noisy_phase, self.target_score

class Discriminator_train_dataset(Dataset):
    def __init__(self, file_list, noisy_path, clean_path):
        self.file_list = file_list
        self.noisy_path = noisy_path
        self.clean_path = clean_path
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx):
        score_filepath = self.file_list[idx]

        enhance_wav, _ = torchaudio.load(score_filepath[-1])
        enhance_mag, _ = get_spec_and_phase(enhance_wav)
        name = os.path.basename(score_filepath[-1])
        if '#' in name:
            name = name.split('#')[0] + '.wav'
            
        noisy_wav, _ = torchaudio.load(self.noisy_path + name)
        noisy_mag, _ = get_spec_and_phase(noisy_wav)

        clean_wav, _ = torchaudio.load(self.clean_path + name)
        clean_mag, _ = get_spec_and_phase(clean_wav)

        True_score = np.asarray(score_filepath[:-1], dtype=np.float32)

        return enhance_mag, noisy_mag, clean_mag, True_score
    
def create_dataloader(filelist, noisy_path, num_target_metric=2, clean_path=None, loader='G', batch_size=1, num_workers=1):
    if loader=='G':
        return DataLoader(dataset=Generator_train_dataset(filelist, noisy_path, num_target_metric),
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          drop_last=True,
                          collate_fn=collate_fn_G)
    elif loader=='D':
        return DataLoader(dataset=Discriminator_train_dataset(filelist, noisy_path, clean_path),
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          drop_last=True,
                          collate_fn=collate_fn_D)
    else:
        raise ValueError("Select G or D loader type!")


def pad_sequence(batch):
    batch = torch.nn.utils.rnn.pad_sequence(batch, padding_value=0.)
    return batch

def collate_fn_G(batch):

    clean_tensors, noise_tensors, targets, lengths = [], [], [], []

    for clean_mag, _, noise_mag, _, target_score in batch:
        clean_tensors += [clean_mag.squeeze(0)]   # clean_mag: [1, T, F] -> [T, F]
        noise_tensors += [noise_mag.squeeze(0)]
        targets += [target_score]
        lengths += [len(clean_mag.squeeze(0))]

    clean_tensors = pad_sequence(clean_tensors).permute(1, 0, 2)   # [T, B, F] -> [B, T, F]
    noise_tensors = pad_sequence(noise_tensors).permute(1, 0, 2)   # [T, B, F] -> [B, T, F]
    targets = torch.stack(targets)                                 # [B, num_target_metric]
    lengths = torch.tensor(lengths)

    return clean_tensors, noise_tensors, targets, lengths


def collate_fn_D(batch):

    enhanced_tensors, noise_tensors, clean_tensors, targets = [], [], [], []

    for enhanced_mag, noise_mag, clean_mag, target_score in batch:
        enhanced_tensors += [enhanced_mag.permute(1, 0, 2)]           # [1, T, F] -> [T, 1, F]
        noise_tensors += [noise_mag.permute(1, 0, 2)]           # [1, T, F] -> [T, 1, F]
        clean_tensors += [clean_mag.permute(1, 0, 2)]           # [1, T, F] -> [T, 1, F]
        targets += [torch.from_numpy(target_score)]

    enhanced_tensors = pad_sequence(enhanced_tensors).permute(1, 2, 0, 3)   # [T, B, 1, F] -> [B, 1, T, F]
    noise_tensors = pad_sequence(noise_tensors).permute(1, 2, 0, 3)   # [T, B, 1, F] -> [B, 1, T, F]
    clean_tensors = pad_sequence(clean_tensors).permute(1, 2, 0, 3)   # [T, B, 1, F] -> [B, 1, T, F]
    targets = torch.stack(targets)                                    # [B, num_target_metric]
    return enhanced_tensors, noise_tensors, clean_tensors, targets