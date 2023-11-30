import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_utils.vocab import NERVocab
import torch
import numpy as np

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, vocab, label_encoder, max_len, batch_size):
        self.data = dataframe
        self.vocab = vocab
        self.label_encoder = label_encoder
        self.max_len = max_len
        self.batch_size = batch_size
        self.X, self.y = self.process_data()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        inputs = torch.tensor(self.X[idx])
        labels = torch.tensor(self.y[idx], dtype=torch.long)
        return {'inputs': inputs, 'labels': labels}

    def process_data(self):
        sentences = self.get_sentences()
        X = [[self.vocab.word_to_idx.get(w[0], self.vocab.word_to_idx['<UNK>']) for w in s] for s in sentences]
        X = pad_sequence([torch.tensor(x, dtype=torch.float32) for x in X], padding_value=float(self.vocab.pad_token_id()), batch_first=True)
        y = [self.label_encoder.transform([w[2] for w in s]) for s in sentences]
        y = pad_sequence([torch.tensor(label, dtype=torch.long) for label in y], padding_value=-1, batch_first=True)

        return X, y

    def get_sentences(self):
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        grouped = self.data.groupby("Sentence #").apply(agg_func)
        return [s for s in grouped]

class Load_data:
    def __init__(self, config):
        self.path = config['path']
        self.max_len = config['max_len']
        self.batch_size = config['batch_size']

    def load_data_(self):
        df = pd.read_csv(self.path, encoding="latin1")
        df = df.fillna(method="ffill")
        vocab = NERVocab(df)
        label_encoder = LabelEncoder()
        all_tags = df['Tag'].unique()
        label_encoder.fit(all_tags)
        max_len = self.max_len
        batch_size = self.batch_size
        dataset = NERDataset(df, vocab, label_encoder, max_len, batch_size)
        train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
        train_data, dev_data = train_test_split(train_data, test_size=0.1, random_state=42)

        return train_data, dev_data, test_data

    def load_data_train_dev_test(self):
        train_data, dev_data, test_data = self.load_data_()
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=self.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, dev_loader, test_loader
