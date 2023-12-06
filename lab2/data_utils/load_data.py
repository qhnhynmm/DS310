import os
from typing import Dict
from datasets import load_dataset
import numpy as np
from torch.utils.data import DataLoader

def loadDataset(config: Dict) -> Dict:
    dataset = load_dataset(
        "csv", 
        data_files={
            "train": os.path.join(config["data"]["dataset_folder"], config["data"]["train_dataset"]),
            "val": os.path.join(config["data"]["dataset_folder"], config["data"]["val_dataset"]),
            "test": os.path.join(config["data"]["dataset_folder"], config["data"]["test_dataset"])
        }
    )
    answer_space = list(np.unique(dataset['train']['sentiment']))
    dataset = dataset.map(
        lambda examples: {'label': [answer_space.index(ans) for ans in examples['sentiment']]},
        batched=True
    )
    
    dataset = dataset.shuffle(123)
    
    train_loader = DataLoader(dataset["train"], batch_size=config["train"]["train_batch_size"], shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=config["train"]["eval_batch_size"], shuffle=False)
    test_loader = DataLoader(dataset["test"], batch_size=config["inference"]["batch_size"], shuffle=False)
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "answer_space": answer_space
    }