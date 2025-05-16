import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerWeatherPredictor(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size):
        super(TransformerWeatherPredictor, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        self.fc = nn.Linear(d_model, output_size)
        self.input_fc = nn.Linear(input_size, d_model)

    def forward(self, x):
        x = self.input_fc(x)
        x = x.permute(1, 0, 2)  # 调整维度以适应Transformer
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # 调整维度回原状
        x = self.fc(x[:, -1, :])  # 取最后一个时间步的输出
        return x