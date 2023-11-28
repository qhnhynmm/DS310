from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def compute_score(labels,preds):
    acc=accuracy_score(labels,preds)
    f1=f1_score(labels,preds,average='macro',zero_division=1)
    precision=precision_score(labels,preds,average='macro',zero_division=1)
    recall=recall_score(labels,preds,average='macro',zero_division=1)
    return acc,f1,precision,recall