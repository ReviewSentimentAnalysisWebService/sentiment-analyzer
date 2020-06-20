import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
        n_layers, bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)

        # pack
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)


        # unpack
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))

        return self.fc(hidden)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)