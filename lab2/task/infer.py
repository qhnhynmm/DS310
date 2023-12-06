import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_utils.load_data import loadDataset

from evaluate.evaluate import compute_score
from tqdm import tqdm
from model.init_model import build_model
class Inference:
    def __init__(self,config):
        self.save_path=os.path.join(config['train']['output_dir'],config['model']['type_model'])
        self.dataloader = loadDataset(config)
        self.answer_space = self.dataloader["answer_space"]     
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = build_model(config,len(self.answer_space)).to(self.device)
    def predict(self):
        test = self.dataloader["test"]
        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'), map_location=self.device)
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print('chưa train model mà đòi test hả')
        self.base_model.eval()
        true_labels = []
        pred_labels = []
        with torch.no_grad():
            for it,item in enumerate(tqdm(test)):
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
                    logits = self.base_model(item['inputs'])
                preds = logits.argmax(axis=-1).cpu().numpy()
                true_labels.extend(item['labels'])
                pred_labels.extend(preds)
        test_acc,test_f1,test_precision,test_recall=compute_score(true_labels,pred_labels,self.Tag_space)
        print(f"test acc: {test_acc:.4f} | test f1: {test_f1:.4f} | test precision: {test_precision:.4f} | test recall: {test_recall:.4f}")
        