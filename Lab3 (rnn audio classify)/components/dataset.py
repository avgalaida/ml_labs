import torch
import torchaudio
from os import listdir
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram


class AudioBinaryClassifyDS(Dataset):
    def __init__(self, dir1, dir2, class_dict):
        super().__init__()

        self.dir1 = dir1
        self.dir2 = dir2
        self.filenames1 = sorted(listdir(dir1))
        self.filenames1.remove('.DS_Store')
        self.filenames2 = sorted(listdir(dir2))
        self.filenames2.remove('.DS_Store')
        self.in_memory_ds = []

        self.class_dict = class_dict

    def pre_comp(self, n_cut):
        for i in range(len(self.filenames1) + len(self.filenames2)):
            if i < len(self.filenames1):
                filename = self.dir1 + self.filenames1[i]
                class_id = torch.tensor(0)
            else:
                filename = self.dir2 + self.filenames2[i-len(self.filenames1)]
                class_id = torch.tensor(1)

            waveform, sample_rate = torchaudio.load(filename, normalize=True)

            if waveform.size(1) >= n_cut*sample_rate:
                wf = waveform[0][:n_cut*sample_rate]
            else:
                pad = torch.zeros(n_cut*sample_rate - waveform.size(1))
                wf = torch.cat((waveform[0], pad))

            transform = MelSpectrogram(sample_rate)
            spectogram = transform(wf.unsqueeze(0))

            self.in_memory_ds.append({'specgram': spectogram[0], 'waveform': wf,
                                      'sr': sample_rate, 'label': class_id})

    def __len__(self):
        return len(self.in_memory_ds)

    def __getitem__(self, idx):
        return self.in_memory_ds[idx]
