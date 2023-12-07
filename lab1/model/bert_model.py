import torch
from torch import nn
from torch.nn import functional as F
from transformers import  AutoModel, AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Optional
from data_utils.vocab import create_ans_space

#design for phobert, xlm-roberta, videberta, bartpho, pretrained in english also supported 
class Text_Embedding(nn.Module):
    def __init__(self, config: Dict):
        super(Text_Embedding,self).__init__()
        self.embedding = AutoModel.from_pretrained(config["text_embedding"]["text_encoder"])
        # freeze all parameters of pretrained model
        if config["text_embedding"]["freeze"]:
            for param in self.embedding.parameters():
                param.requires_grad = False
        self.POS_space,self.Tag_space=create_ans_space(config)
        self.proj = nn.Linear(config["text_embedding"]['d_features'], len(self.Tag_space)+3)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config["text_embedding"]['dropout'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs):
        features = self.embedding(input_ids=inputs).last_hidden_state
        features = self.dropout(self.gelu(features))
        out = self.proj(features)
        return F.log_softmax(out,dim=-1)

class Bert_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = Text_Embedding(config)
        self.POS_space,self.Tag_space=create_ans_space(config)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=len(self.Tag_space))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def forward(self, inputs, labels=None):
        inputs=inputs.to(self.device)
        if labels is not None:
            logits = self.bert(inputs)
            labels=labels.to(self.device)
            loss = self.loss_fn(logits.view(-1,logits.size(-1)), labels.view(-1))
            return logits, loss
        else:
            logits = self.bert(inputs)
            return logits

