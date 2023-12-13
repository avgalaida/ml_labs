import torch.nn as nn
from types import SimpleNamespace


class ResNetBlock(nn.Module):
    def __init__(self, c_in, subsample=False, c_out=-1):
        super().__init__()
        if not subsample:
            c_out = c_in

        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out)
        )

        self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) if subsample else None
        self.act_fn = nn.GELU()

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        out = self.act_fn(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes, num_blocks, c_hidden, dropout_prob=0.5):
        super().__init__()
        self.p = dropout_prob
        self.hparams = SimpleNamespace(
            num_classes=num_classes,
            c_hidden=c_hidden,
            num_blocks=num_blocks,
        )
        self._create_network()

    def _create_network(self):
        self.dropout = nn.Dropout(self.p, inplace=True)
        c_hidden = self.hparams.c_hidden

        self.input_net = nn.Sequential(
            nn.Conv2d(3, c_hidden[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_hidden[0]),
            nn.GELU()
        )

        blocks = []
        for block_idx, block_count in enumerate(self.hparams.num_blocks):
            for bc in range(block_count):
                subsample = bc == 0 and block_idx > 0
                blocks.append(
                    ResNetBlock(
                        c_in=c_hidden[block_idx if not subsample else (block_idx - 1)],
                        subsample=subsample,
                        c_out=c_hidden[block_idx],
                    )
                )
        self.blocks = nn.Sequential(*blocks)

        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(c_hidden[-1], self.hparams.num_classes)
        )

    def forward(self, x):
        x = self.dropout(self.input_net(x))
        x = self.dropout(self.blocks(x))
        x = self.dropout(self.output_net(x))
        return x
