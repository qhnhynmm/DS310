import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_utils.load_data import Load_data
from evaluate.evaluate import compute_score
from model.lstm import LSTM
from tqdm import tqdm

class Classify_Task:
    def __init__(self, config):
        self.num_epochs = config['num_epochs']
        self.patience = config['patience']
        self.learning_rate = config['learning_rate']
        self.best_metric=config['best_metric']
        self.save_path=os.path.join(config['save_path'],config['model'])
        self.dataloader = Load_data(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = LSTM(config).to(self.device)
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.SGD(self.base_model.parameters(),lr=self.learning_rate,momentum=0.5)
    def training(self):
        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)

        train,valid,_ = self.dataloader.load_data_train_dev_test()

        if os.path.exists(os.path.join(self.save_path, 'last_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'last_model.pth'))
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('loaded the last saved model!!!')
            initial_epoch = checkpoint['epoch'] + 1
            print(f"continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("first time training!!!")
            train_loss = 0.

        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'))
            best_score = checkpoint['score']
        else:
            best_score = 0.
            
        threshold=0
        self.base_model.train()
        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            valid_acc = 0.
            valid_f1=0.
            valid_precision=0.
            valid_recall=0.
            train_loss = 0.
            for it,item in enumerate(tqdm(train)):
                inputs, labels = item['inputs'].to(self.device), item['labels'].to(self.device)
                self.optimizer.zero_grad()
                logits,loss = self.base_model(inputs,labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss

            with torch.no_grad():
                for it,item in enumerate(tqdm(valid)):
                    inputs, labels = item['inputs'].to(self.device), item['labels'].to(self.device)
                    logits = self.base_model(inputs)
                    preds = logits.argmax(-1)
                    acc, f1, precision, recall=compute_score(labels.cpu().numpy(),preds.cpu().numpy())
                    valid_acc+=acc
                    valid_f1+=f1
                    valid_precision+=precision
                    valid_recall+=recall
        
            train_loss /= len(train)
            valid_acc /= len(valid)
            valid_f1 /= len(valid)
            valid_precision /= len(valid)
            valid_recall /= len(valid)


            print(f"epoch {epoch + 1}/{self.num_epochs + initial_epoch}")
            print(f"train loss: {train_loss:.4f}")
            print(f"valid acc: {valid_acc:.4f} | valid f1: {valid_f1:.4f} | valid precision: {valid_precision:.4f} | valid recall: {valid_recall:.4f}")

            if self.best_metric =='accuracy':
                score=valid_acc
            if self.best_metric=='f1':
                score=valid_f1
            if self.best_metric=='precision':
                score=valid_precision
            if self.best_metric=='recall':
                score=valid_recall
            # save the last model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.base_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'score': score}, os.path.join(self.save_path, 'last_model.pth'))
            
            # save the best model
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
                    'score':score}, os.path.join(self.save_path, 'best_model.pth'))
                print(f"saved the best model with {self.best_metric} of {score:.4f}")
            
            # early stopping
            if threshold >= self.patience:
                print(f"early stopping after epoch {epoch + 1}")
                break
