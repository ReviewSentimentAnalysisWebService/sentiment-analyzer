import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torchtext.vocab import Vectors
from torchtext.data import TabularDataset, Iterator
import time
import random
from stop_words import stop_words
from konlpy.tag import Mecab
from dnn_models import LSTM


class SentimentAnalyzer():

    def __init__(self):
        SEED = 1234
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mecab = Mecab()
        MAX_VOCAB_SIZE = 8500
        vectors = Vectors(name="cc.ko.100.vec", cache="./")

        self.TEXT = data.Field(
            tokenize=self.tokenizer,
            include_lengths=True,
        )
        self.LABEL = data.LabelField(dtype=torch.float)

        train_data, test_data =  TabularDataset.splits(
            path=".", train="train_dataset.tsv", test="test_dataset.tsv",
            fields=[("text", self.TEXT), ("label", self.LABEL)],
            format="tsv",
            skip_header=True
        )

        self.TEXT.build_vocab(
            train_data,
            max_size = MAX_VOCAB_SIZE,
            vectors=vectors,
            unk_init = torch.Tensor.normal_
        )
        self.LABEL.build_vocab(train_data)

        VOCAB_SIZE = len(self.TEXT.vocab)
        EMBEDDING_DIM = 100
        HIDDEN_DIM = 256
        OUTPUT_DIM = 1
        N_LAYERS = 2
        BIDIRECTIONAL = True
        DROPOUT = 0.2
        PAD_IDX = self.TEXT.vocab.stoi[self.TEXT.pad_token]

        self.model = LSTM(
            VOCAB_SIZE,
            EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT,
            PAD_IDX
        )
        self.model.load_state_dict(torch.load('vocab8500-dr0.2-nlayer2-model-ver2.pt'))

    def tokenizer(self, text: str):
        """
        Custom tokenizer using Mecab
        """
        # remove stopwords defined in stop_words.py
        tokens = [token[0] for token in self.mecab.pos(text, join=False) if token[1] not in stop_words]

        # if the text only has meaningless words, causing tokens empty,
        # append a dummy token in tokens.
        if len(tokens) == 0:
            tokens.append(".")
            
        return tokens

    def analyze(self, sentence: str) -> float:
        self.model.eval()
        tokenized = self.tokenizer(sentence)
        indexed = [self.TEXT.vocab.stoi[token] for token in tokenized]
        length = [len(indexed)]

        tensor = torch.LongTensor(indexed).to(self.device)
        tensor = tensor.unsqueeze(1)  # tensor.shape = seq_len x 1
        length_tensor = torch.LongTensor(length)
        
        prediction = torch.sigmoid(self.model(tensor, length_tensor))

        return prediction.item()