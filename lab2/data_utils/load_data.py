import os
from typing import Dict
from datasets import load_dataset
import numpy as np
from torch.utils.data import DataLoader

class DatasetLoader:
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config['model']['name']
        self.batch_size = config['train']['train_batch_size']
        self.batch_size_test = config['inference']['batch_size']
    def load_dataset(self) -> Dict:
        dataset = load_dataset(
            "csv", 
            data_files={
                "train": os.path.join(self.config["data"]["dataset_folder"], self.config["data"]["train_dataset"]),
                "val": os.path.join(self.config["data"]["dataset_folder"], self.config["data"]["val_dataset"]),
                "test": os.path.join(self.config["data"]["dataset_folder"], self.config["data"]["test_dataset"])
            }
        )

        if self.model_name == "sentiment":
            answer_space = list(np.unique(dataset['train']['sentiment']))
        else:
            answer_space = list(np.unique(dataset['train']['topic']))

        dataset = dataset.map(
            lambda examples: {'label': [answer_space.index(ans) for ans in examples['sentiment']]},
            batched=True
        )

        dataset = dataset.shuffle(123)  # Shuffle the dataset

        # Create DataLoaders
        train_loader = DataLoader(dataset["train"], batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(dataset["val"], batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(dataset["test"], batch_size=self.batch_size_test, shuffle=False)

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "answer_space": answer_space
        }
