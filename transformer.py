import math

import torch
import torch.nn
from torch import Tensor
from torch.nn import (Dropout, Embedding, Linear, LogSoftmax, Module,
                      TransformerEncoder, TransformerEncoderLayer)


class PositionalEncoding(Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [max_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerLangModel(Module):
    def __init__(
        self,
        token_num: int,
        max_len: int,
        d_model: int = 512,
        nhead: int = 8,
    ):
        super().__init__()
        self.token_num = token_num
        self.max_len = max_len
        self.embedding = Embedding(token_num, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=6)
        self.linear = Linear(d_model, token_num)
        self.activation = LogSoftmax()

    def forward(self, src: Tensor) -> Tensor:
        src_embedding = self.positional_encoding(self.embedding(src))
        output = self.transformer_encoder(
            src=src_embedding,
        )
        output = self.activation(self.linear(output)).view(-1, self.token_num)
        return output


class TransformerClassificationModel(Module):
    def __init__(
        self,
        token_num: int,
        max_len: int,
        num_classes: int,
        nhead: int = 8,
        num_encoder_layer: int = 6,
        d_model: int = 512,
    ):
        super().__init__()
        self.max_len = max_len
        self.token_num = token_num
        self.embedding = Embedding(token_num, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers=num_encoder_layer
        )
        if num_classes == 2:
            self.linear = Linear(d_model, num_classes - 1)
        else:
            self.linear = Linear(d_model, num_classes)

    def get_input_feature(self, inputs):
        return self.embedding(inputs)

    def forward_input_feature(self, embeddings):
        output = self.transformer_encoder(
            src=self.positional_encoding(embeddings),
        )
        output = output.mean(dim=0)
        output = self.linear(output)
        return output

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_input_feature(self.get_input_feature(inputs))
