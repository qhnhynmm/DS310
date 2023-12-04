from transformers import AutoModelForTokenClassification, AutoConfig

class NERModel(torch.nn.Module):
    def __init__(self, config):
        super(NERModel, self).__init__()
        self.num_class = config['tokenizer']['num_label']
        self.model_name = config['text_embedding']['model_name']
        config_ = AutoConfig.from_pretrained(self.model_name, num_labels=self.num_class)
        self.bert = AutoModelForTokenClassification.from_pretrained(self.model_name, config=config_)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
