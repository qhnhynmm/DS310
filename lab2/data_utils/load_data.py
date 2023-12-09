import os
from typing import Dict
from datasets import load_dataset
import numpy as np
from torch.utils.data import DataLoader
from data_utils.vocab import create_vocab
from pyvi import ViTokenizer

def loadDataset(config: Dict) -> Dict:
    vocab, _, word_to_index, _ = create_vocab(config)

    index_to_word = {i: w for w, i in word_to_index.items()}

    dataset = load_dataset(
        "csv", 
        data_files={
            "train": os.path.join(config["data"]["dataset_folder"], config["data"]["train_dataset"]),
            "val": os.path.join(config["data"]["dataset_folder"], config["data"]["val_dataset"]),
            "test": os.path.join(config["data"]["dataset_folder"], config["data"]["test_dataset"])
        }
    )

    answer_space_sentiment = list(np.unique(dataset['train']['sentiment']))
    dataset = dataset.map(
        lambda examples: {'label_sentiment': [answer_space_sentiment.index(ans) for ans in examples['sentiment']]},
        batched=True
    )
    answer_space_topic = list(np.unique(dataset['train']['topic']))
    dataset = dataset.map(
        lambda examples: {'label': [answer_space_topic.index(ans) for ans in examples['topic']]},
        batched=True
    )
    for split in dataset.keys():
        dataset[split] = dataset[split].map(
            lambda examples: {'input_ids': [word_to_index.get(word, word_to_index['UNK']) for word in tokenize_sentence(examples['sentence'])]},
            batched=True
        )

    dataset = dataset.shuffle(123)

    # Create DataLoaders
    train_loader = DataLoader(dataset["train"], batch_size=config["train"]["train_batch_size"], shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=config["train"]["eval_batch_size"], shuffle=False)
    test_loader = DataLoader(dataset["test"], batch_size=config["inference"]["batch_size"], shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "answer_space_sentiment": answer_space_sentiment,
        "answer_space_topic":answer_space_topic
    }

def tokenize_sentence(sentence):
    tokenized_sentence = ViTokenizer.tokenize(sentence)
    return tokenized_sentence.split()
