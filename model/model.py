import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import math
import torch.nn as nn
import torch
from Embedding import TransformerEmbedding


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()

        assert d_model % n_head == 0

        self.n_head = n_head
        self.d_model = d_model
        self.attention_head_size = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.n_head, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.transpose_for_scores(q), self.transpose_for_scores(k), self.transpose_for_scores(v)

        attn_score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9)

        attention_probs = nn.functional.softmax(attn_score, dim=-1)
        attention_probs = self.dropout(attention_probs)
        out = torch.matmul(attention_probs, v)

        out = out.permute(0, 2, 1, 3).contiguous()
        new_shape = out.size()[:-2] + (self.d_model,)
        out = out.view(new_shape)
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, hidden)
        self.w_2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, hidden, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(n_head, d_model)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model, hidden, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        _x = x
        x = self.self_attn(x, x, x, mask)
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.layernorm1(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.layernorm2(x + _x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_head, d_model, hidden, vocab_size, embed_size, n_layer, dropout, max_len=5000) -> None:
        super().__init__()
        self.emb = TransformerEmbedding(vocab_size, embed_size, dropout, max_len=max_len)
        self.layer = nn.ModuleList([EncoderLayer(n_head, d_model, hidden, dropout) for _ in range(n_layer)])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layer:
            x = layer(x, src_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, hidden, dropout) -> None:
        super().__init__()
        self.self_attn = MultiHeadedAttention(n_head, d_model)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(d_model)

        self.enc_dec_attn = MultiHeadedAttention(n_head, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model, hidden, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.layernorm3 = nn.LayerNorm(d_model)

    def forward(self, dec, enc, tgt_mask, src_mask):
        _x = dec
        x = self.self_attn(dec, dec, dec, tgt_mask)
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.layernorm1(x + _x)

        if enc is not None:
            _x = x
            x = self.enc_dec_attn(x, enc, enc, src_mask)
            x = self.linear2(x)
            x = self.dropout2(x)
            x = self.layernorm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.layernorm3(x + _x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_head, d_model, hidden, vocab_size, embed_size, n_layer, dropout, max_len) -> None:
        super().__init__()
        self.emb = TransformerEmbedding(vocab_size, embed_size, dropout, max_len=max_len)
        self.layer = nn.ModuleList({DecoderLayer(n_head, d_model, hidden, dropout) for _ in range(n_layer)})
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, dec, enc, tgt_mask, src_mask):
        x = self.emb(dec)
        for layer in self.layer:
            x = layer(x, enc, tgt_mask, src_mask)
        x = self.linear(x)
        return x


class Transformer(nn.Module):
    def __init__(self, n_head, d_model, hidden, vocab_size, embed_size, n_layer, dropout, src_pad_idx, tgt_pad_idx, max_len=5000) -> None:
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        # self.tgt_sos_idx = tgt_sos_idx

        self.encoder = Encoder(n_head, d_model, hidden, vocab_size, embed_size, n_layer, dropout, max_len)
        self.decoder = Decoder(n_head, d_model, hidden, vocab_size, embed_size, n_layer, dropout, max_len)

    def forward(
        self,
        src,
        tgt,
    ):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_trg_mask(tgt)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_src, tgt_mask, src_mask)
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, tgt):
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(3)
        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len)).type(torch.ByteTensor).cuda()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask
