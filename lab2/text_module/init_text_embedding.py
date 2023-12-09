from text_module.pretrained_embedding import Text_Embedding
from text_module.fastext_embedding import Fastext_Embedding

def build_text_embbeding(config):
    if config['text_embedding']['type']=='pretrained':
        return Text_Embedding(config)
    if config['text_embedding']['type']=='fastext':
        return Fastext_Embedding(config)