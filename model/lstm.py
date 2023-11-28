import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout'] 
        self.output_dim = config['output_dim']
        self.vocab_size = config['vocab_size']
        # Thêm lớp nhúng
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # LSTM với lớp nhúng động
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
        
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])
        return output

class LSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LSTMModel = LSTMModel(config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, labels=None):
        if labels is not None:
            logits = self.LSTMModel(inputs)
            loss = self.loss_fn(logits, labels)
            return logits, loss
        else:
            logits = self.LSTMModel(inputs)
            return logits
