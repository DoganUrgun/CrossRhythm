"""EEGNet-8,2 in PyTorch.

Reference:
    V. J. Lawhern et al., "EEGNet: a compact convolutional neural network for
    EEG-based brain-computer interfaces," J. Neural Eng., vol. 15, no. 5, 2018.

This module re-implements the EEGNet model of the authors' Keras code
(github.com/vlawhern/arl-eegmodels, EEGModels.py). The original code is
distributed under the U.S. Army Research Laboratory license; see
THIRD_PARTY_NOTICES.md. EEGNet is used both as a competing decoder and as the
compact pathway of CrossRhythm.

Input: EEG of shape [batch, channels, samples] at 250 Hz.
Output: unnormalized class logits [batch, classes].
"""

import math

import torch
from torch import nn


class EEGNet(nn.Module):
    def __init__(self, channels, samples, n_classes, sampling_rate=250, resampling_rate=128,
                 temporal_filters=8, depth_multiplier=2, dropout=0.25,
                 spatial_max_norm=1.0, classifier_max_norm=0.25):
        super().__init__()
        self.channels, self.samples = int(channels), int(samples)
        f1, d = int(temporal_filters), int(depth_multiplier)
        f2 = f1 * d

        # EEGNet was designed for 128 Hz input, so the signal is resampled internally.
        if sampling_rate == resampling_rate:
            self.resample = nn.Identity()
        else:
            from torchaudio.transforms import Resample
            self.resample = Resample(orig_freq=sampling_rate, new_freq=resampling_rate)
        resampled_samples = (samples * resampling_rate + sampling_rate - 1) // sampling_rate

        # Kernel lengths of the original model: 1/4 s (temporal) and 1/8 s (separable).
        kernel_length = round(resampling_rate / 4)
        separable_length = round(resampling_rate / 8)
        self.spatial_max_norm = float(spatial_max_norm)
        self.classifier_max_norm = float(classifier_max_norm)

        # Batch normalization uses the Keras defaults (epsilon 1e-3, momentum 0.99).
        self.temporal = nn.Conv2d(1, f1, (1, kernel_length), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(f1, eps=1e-3, momentum=0.01)
        self.spatial = nn.Conv2d(f1, f1 * d, (channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f1 * d, eps=1e-3, momentum=0.01)
        self.activation = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        self.separable_depthwise = nn.Conv2d(f1 * d, f1 * d, (1, separable_length),
                                             groups=f1 * d, padding="same", bias=False)
        self.separable_pointwise = nn.Conv2d(f1 * d, f2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(f2, eps=1e-3, momentum=0.01)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        self.classifier = nn.Linear(f2 * (resampled_samples // 4 // 8), n_classes)

        # Glorot-uniform initialization as in Keras.
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Keras computes the fans of a depthwise kernel from its [C, 1, F1, D] layout.
        bound = math.sqrt(6.0 / (channels * (f1 + d)))
        nn.init.uniform_(self.spatial.weight, -bound, bound)

    def forward(self, x):
        x = self.bn1(self.temporal(self.resample(x.contiguous()).unsqueeze(1)))
        x = self.activation(self.bn2(self.spatial(x)))
        x = self.dropout1(self.pool1(x))
        x = self.separable_pointwise(self.separable_depthwise(x))
        x = self.dropout2(self.pool2(self.activation(self.bn3(x))))
        # Flatten in the Keras channels-last order (time before feature maps).
        x = x.permute(0, 2, 3, 1).flatten(1)
        return self.classifier(x)

    @torch.no_grad()
    def project_weights(self):
        """Max-norm constraints of the original model, applied after each optimizer step."""
        for weight, limit in ((self.spatial.weight, self.spatial_max_norm),
                              (self.classifier.weight, self.classifier_max_norm)):
            norms = weight.flatten(1).norm(p=2, dim=1)
            scale = (limit / norms.clamp_min(1e-12)).clamp(max=1.0)
            weight.mul_(scale.reshape((-1,) + (1,) * (weight.ndim - 1)))
