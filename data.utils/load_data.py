import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from vocab import NERVocab
import torch

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, vocab, label_encoder, max_len):
        self.data = dataframe
        self.vocab = vocab
        self.label_encoder = label_encoder
        self.max_len = max_len
        self.X, self.y = self.process_data()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        inputs = torch.tensor(self.X[idx])
        labels = torch.tensor(self.y[idx], dtype=torch.long)
        return inputs, labels

    def process_data(self):
        sentences = self.get_sentences()
        X = [[self.vocab.word_to_idx.get(w[0], self.vocab.word_to_idx['<UNK>']) for w in s] for s in sentences]
        X = pad_sequence([torch.tensor(x) for x in X], padding_value=self.vocab.pad_token_id, batch_first=True)

        y = [self.label_encoder.transform([w[2] for w in s]) for s in sentences]

        return X, y

    def get_sentences(self):
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        grouped = self.data.groupby("Sentence #").apply(agg_func)
        return [s for s in grouped]

def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    vocab = NERVocab(df)
    label_encoder = LabelEncoder()
    df['Tag'] = label_encoder.fit_transform(df['Tag'])

    # Split the data into training, dev, and test sets
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    train_data, dev_data = train_test_split(train_data, test_size=0.1, random_state=42)

    max_len = 50  # You can adjust this value based on your data
    train_dataset = NERDataset(train_data, vocab, label_encoder, max_len)
    dev_dataset = NERDataset(dev_data, vocab, label_encoder, max_len)
    test_dataset = NERDataset(test_data, vocab, label_encoder, max_len)

    return train_dataset, dev_dataset, test_dataset
