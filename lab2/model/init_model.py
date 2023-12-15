
from model.bert_cnn import createTextCNN_Model
from model.classify import createclassify
def build_model(config, answer_space):
    if config['model']['type_model']=='bert_cnn':
        return createTextCNN_Model(config,answer_space)
    if config['model']['type_model']=='classify':
        return createclassify(config,answer_space)