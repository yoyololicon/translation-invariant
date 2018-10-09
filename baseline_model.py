import torch
from torch import nn
import numpy as np


def create_filters(d, k, sr=44100, start_freq=50, end_freq=6000):
    x = torch.linspace(0, 2 * np.pi, d + 1)[:d]
    num_cycles = start_freq * d / sr
    scaling_ind = np.log(end_freq / start_freq) / k
    window_mask = 1 - torch.cos(x)

    phases = torch.exp(torch.arange(float(k)) * scaling_ind)[None, :] * num_cycles * x[:, None]
    filters = torch.stack((torch.cos(phases), torch.sin(phases)), 2) * window_mask[:, None, None]
    return filters


class spectrograms(nn.Module):
    k = 512
    d = 4096

    d2_x = 1  # lvl2 input dims_x
    d2_y = 128  # lvl2 input dims_y
    k2 = 128  # num lvl2 filters
    stride_y = 2  # lvl2 stride

    d3_x = 25  # lvl3 input dims_x
    d3_y = 1  # lvl3 input dims_y (fully connected)
    k3 = 256  # num lvl3 filters

    window_size = 16384
    hop_size = 512
    num_regions = 1 + (window_size - d) // hop_size  # 25
    num_regions2_x = num_regions
    num_regions2_y = 1 + (k - d2_y) // stride_y  # 193
    num_regions3_x = 1 + (num_regions2_x - d3_x) // 1  # 1
    num_regions3_y = num_regions2_y  # 193

    m = 128

    def __init__(self):
        super().__init__()
        # construct sin/cos weights
        self.filter_banks = nn.Parameter(create_filters(self.d, self.k), requires_grad=False)

        # need moving average?
        self.w_w2 = nn.Sequential(nn.Conv2d(1, self.k2, (self.d2_x, self.d2_y), (1, self.stride_y), bias=False),
                                  nn.ReLU(),
                                  nn.Conv2d(self.k2, self.k3, (self.d3_x, self.d3_y), bias=False),
                                  nn.ReLU())
        self.beta = nn.Linear(self.num_regions3_x * self.num_regions3_y * self.k3, self.m, bias=False)

    def forward(self, x):
        # batch x 16384
        x = x.unfold(1, self.d, self.hop_size) @ self.filter_banks.view(self.d, self.k * 2)
        zx = x.view(-1, self.num_regions, self.k, 2).pow(2).sum(3).unsqueeze(1)
        # batch x 1 x 25 x 512
        z3 = self.w_w2(torch.log(zx + 10e-15))
        # batch x 256 x 1 x 193
        y = self.beta(z3.view(-1, self.num_regions3_x * self.num_regions3_y * self.k3))
        return y


class EMA(nn.Module):
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


if __name__ == '__main__':
    import librosa

    y, sr = librosa.load(librosa.util.example_audio_file())
    w = create_filters(2048, 128, sr)
    y = torch.from_numpy(y).float()

    y = y.unfold(0, 2048, 512) @ w.view(2048, 256)
    y = y.view(-1, 128, 2).pow(2).sum(2)
    y = y.detach().numpy().T
    print(y.shape)

    from matplotlib import pyplot as plt

    plt.imshow(np.log1p(y), aspect='auto', origin='lower')
    plt.show()
