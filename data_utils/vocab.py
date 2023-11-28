class NERVocab:
    def __init__(self, data):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.build_vocab(data)

    def build_vocab(self, data):
        unique_words = set(data['Word'].tolist())
        self.word_to_idx = {word: idx + 1 for idx, word in enumerate(unique_words)}
        self.word_to_idx['<UNK>'] = 0  # Add <UNK> token
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

    def convert_tokens_to_ids(self, tokens):
        return [self.word_to_idx.get(token, 0) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.idx_to_word[idx] for idx in ids]

    def vocab_size(self):
        return len(self.word_to_idx)

    def pad_token_id(self):
        return 0  # ID for the padding token
