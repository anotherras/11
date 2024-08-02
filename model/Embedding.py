import torch.nn as nn
import math
import torch


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)].requires_grad_(False)


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout, max_len=5000) -> None:
        super().__init__()
        self.embedding = Embeddings(vocab_size, embed_size)
        self.pos_embedding = PositionalEmbedding(embed_size, max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.embedding(x) + self.pos_embedding(x))
