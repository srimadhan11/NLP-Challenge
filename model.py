import math

import torch
import torch.nn as nn


# Function that computes PositionalEncoding vector
def get_PE(max_len, d_model):
    PE         = torch.empty(max_len, d_model, dtype=torch.float32)
    position   = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    multiplier = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    angle      = position * multiplier

    PE[:, 0::2] = torch.sin(angle)
    PE[:, 1::2] = torch.cos(angle[:, :d_model//2])
    PE          = PE.unsqueeze(0).transpose(0, 1)
    return PE


# nn.Module Transformer
class Transformer(nn.Module):
    def __init__(self, embedding_dim, src_pad_idx, nhead,
                     src_vocab_size, tgt_vocab_size,
                     num_encoder_layers, num_decoder_layers,
                     max_src_seq_len, max_tgt_seq_len,
                     dim_feedforward, dropout_p):

        super(Transformer, self).__init__()

        self.PE = get_PE(max(max_src_seq_len, max_tgt_seq_len), embedding_dim)

        self.src_word_embedding = nn.Embedding(src_vocab_size , embedding_dim)
        self.tgt_word_embedding = nn.Embedding(tgt_vocab_size , embedding_dim)

        self.transformer = nn.Transformer(embedding_dim, nhead,
                                          num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, dropout_p)

        self.fnn_out = nn.Linear(embedding_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout_p)

        self.max_src_seq_len = max_src_seq_len
        self.max_tgt_seq_len = max_tgt_seq_len
        self.embedding_dim   = embedding_dim
        self.src_pad_idx     = src_pad_idx
        pass


    def forward(self, src, tgt):
        src, tgt = src.T, tgt.T

        src_seq_length, src_N = src.shape
        tgt_seq_length, tgt_N = tgt.shape
        assert src_seq_length <= self.max_src_seq_len and tgt_seq_length <= self.max_tgt_seq_len

        src_PE = self.PE[:src_seq_length, :].to(src.device)
        tgt_PE = self.PE[:tgt_seq_length, :].to(tgt.device)

        embed_src = self.dropout(self.src_word_embedding(src) + src_PE)
        embed_tgt = self.dropout(self.tgt_word_embedding(tgt) + tgt_PE)

        src_mask = (src.transpose(0, 1) == self.src_pad_idx)      # shape: (N, src_seq_length)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_length).to(tgt.device)

        del src
        del tgt

        out = self.transformer(embed_src, embed_tgt, src_key_padding_mask=src_mask, tgt_mask=tgt_mask)

        del src_mask
        del tgt_mask
        del embed_src
        del embed_tgt

        out = out.transpose(0, 1)
        out = self.fnn_out(out)
        return out