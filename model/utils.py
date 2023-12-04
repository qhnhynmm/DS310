import torch

def generate_padding_mask(sequences, padding_idx: int) -> torch.BoolTensor:
    '''
        sequences: (bs, seq_len) or (bs, seq_len, dim)
    '''
    if sequences is None:
        return None

    if len(sequences.shape) == 2:  # (bs, seq_len)
        __seq = sequences.unsqueeze(dim=-1)  # (bs, seq_len, 1)
    else: # (bs, deq_len, dim)
        __seq = sequences

    mask = (torch.sum(__seq, dim=-1) == padding_idx)  # (b_s, seq_len)
    return mask

def generate_sequential_mask(seq_len: int) -> torch.BoolTensor:
    '''
        Mask out subsequent positions
    '''
    attn_shape = (seq_len, seq_len)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).to(torch.bool)

    return subsequent_mask