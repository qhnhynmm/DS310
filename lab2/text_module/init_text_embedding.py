# from text_module.count_vectorizer import CountVectorizer
# from text_module.tf_idf import IDFVectorizer
from data_utils.vocab import create_vocab
from text_module.embedding import Text_Embedding

def build_text_embbeding(config):
    vocab,word_count=create_vocab(config)
    if config['text_embedding']['type']=='pretrained':
        return Text_Embedding(config)
    # if config['text_embedding']['type']=='count_vector':
    #     return CountVectorizer(config,vocab)
    # if config['text_embedding']['type']=='tf_idf':
    #     return IDFVectorizer(config,vocab,word_count)