from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.cnn import Text_CNN
from text_module.init_text_embedding import build_text_embbeding

class TextCNN_Model(nn.Module):
    def __init__(self,config: Dict, num_labels: int):
     
        super(TextCNN_Model, self).__init__()
        self.num_labels = num_labels
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout=config["model"]["dropout"]
        self.d_text = config["text_embedding"]['d_features']
        self.max_length = config['tokenizer']['max_length']
        self.embed_type=config['text_embedding']['type']
        self.text_embbeding = build_text_embbeding(config)
        self.max_length = config["tokenizer"]["max_length"]
        self.classifier = Text_CNN(self.intermediate_dims,self.num_labels)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text: List[str], labels: Optional[torch.LongTensor] = None):
        if self.embed_type not in ['count_vector','tf_idf']:
            embbed, mask = self.text_embbeding(text)
        else:
            embbed=self.text_embbeding(text)
            mask=None
        logits = self.classifier(embbed)
        logits = F.log_softmax(logits, dim=-1)
        out = {
            "logits": logits
        }
        if labels is not None:
            # logits=logits.view(-1,self.num_labels)
            # labels = labels.view(-1)
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        
        return out


def createTextCNN_Model(config: Dict, answer_space: List[str]) -> TextCNN_Model:
    return TextCNN_Model(config, num_labels=len(answer_space))