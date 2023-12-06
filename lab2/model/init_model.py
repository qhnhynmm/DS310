
from model.bert_cnn import createTextCNN_Model,TextCNN_Model

def build_model(config, answer_space):
    if config['model']['type_model']=='bert_cnn':
        return createTextCNN_Model(config,answer_space)
    
def get_model(config, num_labels):
    if config['model']['type_model']=='bert_cnn':
        return TextCNN_Model(config,num_labels)