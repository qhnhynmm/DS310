from typing import Dict
from collections import Counter
from datasets import load_dataset
import os
from pyvi import ViTokenizer

def tokenize_sentence(sentence):
    tokenized_sentence = ViTokenizer.tokenize(sentence)
    return tokenized_sentence.split()

def create_vocab(config: Dict):
    data_folder = config['data']['dataset_folder']
    train_set = config["data"]["train_dataset"]
    val_set = config["data"]["val_dataset"]
    test_set = config["data"]["test_dataset"]

    dataset = load_dataset(
        "csv", 
        data_files={
            "train": os.path.join(data_folder, train_set),
            "val": os.path.join(data_folder, val_set),
            "test": os.path.join(data_folder, test_set)
        }
    )

    word_counts = Counter()

    for data_file in dataset.values():
        try:
            for item in data_file['sentence']:
                # Tokenize the sentence using pyvi
                words = tokenize_sentence(item)
                word_counts.update(words)
        except Exception as e:
            print(f"Error processing data file: {e}")

    word_counts['UNK'] = len(word_counts) + 2
    word_counts['PAD'] = len(word_counts) + 1

    word_to_index = {w: i + 2 for i, w in enumerate(word_counts)}
    word_to_index['UNK'] = 1
    word_to_index['PAD'] = 0

    index_to_word = {i: w for w, i in word_to_index.items()}

    vocab = list(word_counts.keys())

    return vocab, word_counts, word_to_index, index_to_word
