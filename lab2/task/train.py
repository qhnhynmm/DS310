import torch
import torch.optim as optim
import os
from data_utils.load_data import loadDataset
from evaluate.evaluate import compute_score
from tqdm import tqdm
from model.init_model import createTextCNN_Model

class Classify_Task:
    def __init__(self, config):
        self.num_epochs = config['train']['num_train_epochs']
        self.patience = config['train']['patience']
        self.learning_rate = config['train']['learning_rate']
        self.best_metric = config['train']['metric_for_best_model']
        self.save_path = os.path.join(config['train']['output_dir'], config['model']['type_model'])
        self.weight_decay = config['train']['weight_decay']
        self.dataloader = loadDataset(config)
        self.answer_space = self.dataloader["answer_space"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = createTextCNN_Model(config, self.answer_space).to(self.device)
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler()
        lambda1 = lambda epoch: 0.95 ** epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

    def training(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        train_loader = self.dataloader["train_loader"]
        val_loader = self.dataloader["val_loader"]

        if os.path.exists(os.path.join(self.save_path, 'last_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'last_model.pth'))
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded the last saved model!!!')
            initial_epoch = checkpoint['epoch'] + 1
            print(f"Continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("First time training!!!")

        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'))
            best_score = checkpoint['score']
        else:
            best_score = 0.

        threshold = 0
        self.base_model.train()

        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            valid_acc = 0.
            valid_f1 = 0.
            valid_precision = 0.
            valid_recall = 0.
            train_loss = 0.

            for it, item in enumerate(tqdm(train_loader)):
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
                    # Assuming item['sentence'] is a list of strings
                    inputs = item['sentence']
                    labels = item['label'].to(self.device)
                    outputs = self.base_model(inputs, labels)

                logits = outputs["logits"]
                loss = outputs.get("loss", None)
                if loss is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    train_loss += loss.item()

            with torch.no_grad():
                for it, item in enumerate(tqdm(val_loader)):
                    with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
                        inputs = item['sentence']
                        logits = self.base_model(inputs)

                    preds = logits["logits"].argmax(-1)

                    acc, f1, precision, recall = compute_score(item['label'].cpu().numpy(), preds.cpu().numpy())
                    valid_acc += acc
                    valid_f1 += f1
                    valid_precision += precision
                    valid_recall += recall

            train_loss /= len(train_loader)
            valid_acc /= len(val_loader)
            valid_f1 /= len(val_loader)
            valid_precision /= len(val_loader)
            valid_recall /= len(val_loader)

            print(f"Epoch {epoch + 1}/{self.num_epochs + initial_epoch}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Valid Acc: {valid_acc:.4f} | Valid F1: {valid_f1:.4f} | Valid Precision: {valid_precision:.4f} | Valid Recall: {valid_recall:.4f}")

            if self.best_metric == 'accuracy':
                score = valid_acc
            elif self.best_metric == 'f1':
                score = valid_f1
            elif self.best_metric == 'precision':
                score = valid_precision
            elif self.best_metric == 'recall':
                score = valid_recall

            # Save the last model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.base_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'score': score
            }, os.path.join(self.save_path, 'last_model.pth'))

            # Save the best model
            if epoch > 0 and score < best_score:
                threshold += 1
            else:
                threshold = 0

            if score > best_score:
                best_score = score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.base_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'score': score
                }, os.path.join(self.save_path, 'best_model.pth'))
                print(f"Saved the best model with {self.best_metric} of {score:.4f}")

            # Early stopping
            if threshold >= self.patience:
                print(f"Early stopping after epoch {epoch + 1}")
                break
