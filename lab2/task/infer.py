import torch
from tqdm import tqdm
from data_utils.load_data import loadDataset
from evaluate.evaluate import compute_score
from model.init_model import build_model
import os
class Inference:
    def __init__(self, config):
        self.save_path = os.path.join(config['train']['output_dir'], config['model']['type_model'])
        self.dataloader = loadDataset(config)
        self.answer_space = self.dataloader["answer_space"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = build_model(config, self.answer_space).to(self.device)

    def predict(self):
        test_loader = self.dataloader["test_loader"]
        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'), map_location=self.device)
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print('Model has not been trained yet.')
            return

        self.base_model.eval()
        true_labels = []
        pred_labels = []

        with torch.no_grad():
            for it, item in enumerate(tqdm(test_loader)):
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
                        inputs = item['sentence']
                        logits = self.base_model(inputs)
                
                preds = logits["logits"].argmax(-1)
                true_labels.extend(item['label'])
                pred_labels.extend(preds)

        acc, f1, precision, recall = compute_score(item['label'].cpu().numpy(), preds.cpu().numpy())
        print(f"Test Accuracy: {acc:.4f} | Test F1: {f1:.4f} | Test Precision: {precision:.4f} | Test Recall: {recall:.4f}")

