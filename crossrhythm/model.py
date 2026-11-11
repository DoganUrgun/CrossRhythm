"""CrossRhythm: a multi-pathway decoder for motor imagery EEG.

The network has four pathways whose logits are averaged at inference:

1. Rhythm pathway   learnable sinc band-pass filters, spatial filters per band and
                    log-variance (band-power) features.
2. Residual pathway multiscale convolutional stem followed by residual blocks.
3. Transformer pathway the same stem followed by a transformer encoder.
4. Compact pathway  EEGNet-8,2 applied to the same input.

The multiscale stem, its auxiliary per-scale heads and the transformer layout
follow MSVTNet (K. Liu et al., IEEE J. Biomed. Health Inform., 2024). The
windowed-sinc filters follow SincNet (M. Ravanelli and Y. Bengio, IEEE SLT, 2018).

Input: aligned and standardized EEG [batch, channels, samples].
"""

import torch
from torch import nn
from torch.nn import functional as F

from .eegnet import EEGNet


def temporal_bin_means(x, bins=4):
    """Average the last axis over `bins` intervals (same intervals as AdaptiveAvgPool1d)."""
    length = x.shape[-1]
    return torch.stack([
        x[..., index * length // bins:((index + 1) * length + bins - 1) // bins].mean(-1)
        for index in range(bins)], dim=-1)


class TemporalResidualBlock(nn.Module):
    """Depthwise temporal convolution followed by pointwise feature mixing."""

    def __init__(self, width, dropout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(width, width, 9, padding=4, groups=width, bias=False),
            nn.BatchNorm1d(width), nn.ELU(),
            nn.Conv1d(width, 2 * width, 1, bias=False),
            nn.BatchNorm1d(2 * width), nn.ELU(), nn.Dropout(dropout),
            nn.Conv1d(2 * width, width, 1, bias=False),
            nn.BatchNorm1d(width))

    def forward(self, x):
        return F.elu(x + self.body(x))


class RhythmPathway(nn.Module):
    """Learnable band-pass filters -> spatial filters per band -> log-variance features.

    Ten sinc filters are initialized to the bands below and constrained to 4-40 Hz
    with a bandwidth of at least 1 Hz. Each band has two spatial filters, and the
    log-variance is computed in four equal time windows (10 x 2 x 4 = 80 features).
    """

    INITIAL_BANDS = ((4, 8), (8, 10), (10, 12), (12, 14), (14, 16),
                     (16, 20), (20, 24), (24, 28), (28, 34), (34, 40))

    def __init__(self, channels, samples, n_classes, kernel=125, spatial_filters=2,
                 sampling_rate=250, windows=4):
        super().__init__()
        bands = len(self.INITIAL_BANDS)
        self.samples, self.kernel, self.windows = int(samples), int(kernel), int(windows)
        self.sampling_rate = float(sampling_rate)
        self.features = bands * spatial_filters * windows

        # Cutoffs are parameterized through sigmoids, which keeps them inside 4-40 Hz.
        initial = torch.tensor(self.INITIAL_BANDS, dtype=torch.float32)
        low = initial[:, 0]
        self.low_logits = nn.Parameter(torch.logit(((low - 4) / 35).clamp(1e-6, 1 - 1e-6)))
        width_fraction = (initial[:, 1] - low - 1) / (39 - low)
        self.width_logits = nn.Parameter(torch.logit(width_fraction.clamp(1e-6, 1 - 1e-6)))
        self.register_buffer("filter_time", torch.arange(kernel, dtype=torch.float32) - kernel // 2)
        self.register_buffer("filter_window", torch.hamming_window(kernel, periodic=False))

        self.spatial = nn.Conv2d(bands, bands * spatial_filters, (channels, 1), groups=bands, bias=False)
        self.feature_norm = nn.LayerNorm(self.features)
        self.head = nn.Linear(self.features, n_classes)

    def band_cutoffs_hz(self):
        """Current lower and upper cutoff of every band, in Hz."""
        low = 4 + 35 * self.low_logits.sigmoid()
        high = low + 1 + (39 - low) * self.width_logits.sigmoid()
        return torch.stack((low, high), dim=1)

    def temporal_filters(self):
        """Hamming-windowed sinc band-pass filters with unit energy."""
        cutoffs = self.band_cutoffs_hz() / self.sampling_rate
        low, high = cutoffs[:, :1], cutoffs[:, 1:]
        time = self.filter_time.unsqueeze(0)
        impulse = 2 * high * torch.sinc(2 * high * time) - 2 * low * torch.sinc(2 * low * time)
        impulse = impulse * self.filter_window
        impulse = impulse / impulse.square().sum(1, keepdim=True).sqrt().clamp_min(1e-12)
        return impulse[:, None, None, :]

    def forward(self, x):
        # Reflection padding uses only samples of the current trial.
        padded = F.pad(x.unsqueeze(1), (self.kernel // 2, self.kernel // 2, 0, 0), mode="reflect")
        filtered = F.conv2d(padded, self.temporal_filters())
        spatial = self.spatial(filtered).squeeze(2)
        step = self.samples // self.windows
        log_variances = torch.stack([
            spatial[..., index * step:(index + 1) * step].var(dim=-1, unbiased=False).clamp_min(1e-6).log()
            for index in range(self.windows)], dim=-1)
        features = self.feature_norm(log_variances.flatten(1))
        return self.head(features)


class CrossRhythm(nn.Module):
    """Four-pathway decoder; see the module docstring.

    Args:
        channels, samples, n_classes: input and output dimensions.
        width: number of stem features D (the paper uses 80, i.e. F = D / 8 = 10
            filters per kernel length).
        rhythm_pathway: set to False to train the variant without the rhythm pathway.
    """

    def __init__(self, channels, samples, n_classes, width=80, rhythm_pathway=True,
                 convolution_dropout=0.1, transformer_dropout=0.3, residual_dropout=0.25,
                 transformer_layers=2):
        super().__init__()
        self.channels, self.samples = int(channels), int(samples)
        filters, scale_width = width // 8, width // 4
        tokens = samples // 8 // 7

        # Multiscale stem: temporal convolutions with four kernel lengths, each followed
        # by a depthwise spatial convolution over all channels.
        self.stems = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, filters, (1, kernel), padding="same", bias=False),
                nn.BatchNorm2d(filters),
                nn.Conv2d(filters, scale_width, (channels, 1), groups=filters, bias=False),
                nn.BatchNorm2d(scale_width))
            for kernel in (15, 31, 63, 125)])
        self.first_pool = nn.Sequential(nn.ELU(), nn.AvgPool1d(8), nn.Dropout(convolution_dropout))

        # Per-scale temporal convolutions that produce the transformer tokens, and
        # auxiliary classification heads (one per scale) used only during training.
        self.scale_temporal = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(scale_width, scale_width, 15, padding=7, groups=scale_width, bias=False),
                nn.BatchNorm1d(scale_width), nn.ELU(),
                nn.AvgPool1d(7), nn.Dropout(convolution_dropout))
            for _ in range(4)])
        self.scale_heads = nn.ModuleList([nn.Linear(tokens * scale_width, n_classes) for _ in range(4)])

        # Residual pathway.
        self.conv_body = nn.Sequential(TemporalResidualBlock(width, residual_dropout),
                                       TemporalResidualBlock(width, residual_dropout))
        self.conv_head = nn.Linear(4 * width, n_classes)

        # Transformer pathway (class token, learnable positions, pre-norm layers).
        self.class_token = nn.Parameter(torch.zeros(1, 1, width))
        self.position_embedding = nn.Parameter(torch.zeros(1, tokens + 1, width))
        self.embedding_dropout = nn.Dropout(transformer_dropout)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=8, dim_feedforward=2 * width,
                                           dropout=transformer_dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers,
                                                 norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.transformer_head = nn.Linear(width, n_classes)

        # Compact pathway.
        self.compact = EEGNet(channels=channels, samples=samples, n_classes=n_classes)

        # Rhythm pathway.
        self.rhythm = RhythmPathway(channels, samples, n_classes) if rhythm_pathway else None

    def forward_with_aux(self, x):
        """Return (averaged logits, logits of each pathway, auxiliary per-scale logits)."""
        responses = [stem(x.unsqueeze(1)).squeeze(2) for stem in self.stems]
        fine = [self.first_pool(response) for response in responses]

        residual = self.conv_body(torch.cat(fine, dim=1))
        pathways = [self.conv_head(temporal_bin_means(residual).flatten(1))]

        scales = [block(response) for block, response in zip(self.scale_temporal, fine)]
        auxiliary = tuple(head(response.flatten(1)) for head, response in zip(self.scale_heads, scales))
        sequence = torch.cat(scales, dim=1).transpose(1, 2)
        sequence = torch.cat((self.class_token.expand(len(x), -1, -1), sequence), dim=1)
        sequence = self.embedding_dropout(sequence + self.position_embedding)
        pathways.append(self.transformer_head(self.transformer(sequence)[:, 0]))

        pathways.append(self.compact(x))
        if self.rhythm is not None:
            pathways.append(self.rhythm(x))
        return torch.stack(pathways).mean(0), tuple(pathways), auxiliary

    def forward(self, x):
        return self.forward_with_aux(x)[0]

    def training_loss(self, x, labels, label_smoothing=0.1, pathway_weight=0.6):
        """Training loss: weighted mean of the pathway losses plus the auxiliary per-scale losses."""
        logits, pathways, auxiliary = self.forward_with_aux(x)

        def cross_entropy(head):
            return F.cross_entropy(head, labels, label_smoothing=label_smoothing)

        pathway_loss = torch.stack([cross_entropy(head) for head in pathways]).mean()
        auxiliary_loss = sum((cross_entropy(head) for head in auxiliary), logits.new_zeros(()))
        return pathway_weight * pathway_loss + (1 - pathway_weight) * auxiliary_loss, logits

    @torch.no_grad()
    def project_weights(self):
        self.compact.project_weights()
