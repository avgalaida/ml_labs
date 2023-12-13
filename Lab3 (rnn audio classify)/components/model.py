import torch
import torch.nn as nn


class AudioRNNBinary(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, drop_out=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_out)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        )

    def forward(self, x):
        batch_size = x.size(0)
        h0, c0 = self.init_hidden(batch_size, device=x.device)
        ula, (h_out, _) = self.lstm(x, (h0, c0))
        h_out = h_out[-1]
        out = self.fc(h_out)

        return out.squeeze(dim=1)


