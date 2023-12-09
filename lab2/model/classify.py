from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.init_text_embedding import build_text_embbeding

class classify(nn.Module):
    def __init__(self,config: Dict, num_labels: int):
     
        super(classify, self).__init__()
        self.num_labels = num_labels
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout=config["model"]["dropout"]
        self.d_text = config["text_embedding"]['d_features']

        self.text_embbeding = build_text_embbeding(config)
        self.classifer = nn.Linear(self.intermediate_dims, num_labels)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text: List[str], labels: Optional[torch.LongTensor] = None):
        embbed = self.text_embbeding(text)
        mean_pooling = torch.mean(embbed, dim=1)
        logits = self.classifer(mean_pooling)
        logits = F.log_softmax(logits, dim=-1)
        out = {
            "logits": logits
        }
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        
        return out
def createclassify(config: Dict, answer_space: List[str]) -> classify:
    return classify(config, num_labels=len(answer_space))