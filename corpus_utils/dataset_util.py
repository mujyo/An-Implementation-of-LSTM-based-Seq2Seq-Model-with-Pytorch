import random

from torch.utils.data import Dataset
import torch

"""
The implementation of collcate refers to https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/13
and https://github.com/yunjey/seq2seq-dataloader/blob/master/data_loader.py
Custom Dataset refers to https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py
"""
class PolyExpDataset(Dataset):
    """
    Custom Dataset for polynomial Expansion
    """
    def __init__(self, src_seqs, tgt_seqs, vocab):
        self.src_seqs = src_seqs
        self.tgt_seqs = tgt_seqs
        self.vocab = vocab

    def __len__(self):
        return len(self.src_seqs)

    def __getitem__(self, index):
        src_seq = self.src_seqs[index]
        tgt_seq = self.tgt_seqs[index]
        factorized_src = self.vocab.transform_to_index(src_seq) + [self.vocab.EOS]
        expanded_tgt = self.vocab.transform_to_index(tgt_seq) + [self.vocab.EOS]
        
        return[torch.tensor(factorized_src), torch.tensor(expanded_tgt)]

def collate_fn(src_tgt_seqs):
    """
    Create minibatch tensor from (input, output) lists
    Args:
        src_tgt_seqs: list of tuples(input, output)
        - input: torch tensor of shape (); variable length
        - output: torch tensor of shape(); variable length
    Return:
        input: torch tensor of shape (batch_size, padded_length)
        input_lens: list of length (batch_size); valid length for each padded source sequence
        output: torch tensor of shape (batch_size, padded_length)
        trg_lengths: list of length (batch_size); valid length for each padded target sequence
    """
    def make_minibatch(seqs):
        batch_size = len(seqs)
        lens = [len(s) for s in seqs]
        max_lens = max(lens)
        padded_seqs = torch.zeros(batch_size, max_lens).long()
        for i, s in enumerate(seqs):
            end = lens[i]
            padded_seqs[i, :end] = s[:end]
        return padded_seqs, lens
    
    # sort a list by sequence length by descending order to enable pack_padded_sequence
    src_tgt_seqs.sort(key=lambda x:len(x[0]), reverse=True)
    src_seqs, tgt_seqs = zip(*src_tgt_seqs)

    # pad sequences with in the minibatch into the same length
    src_seqs, src_lens = make_minibatch(src_seqs)
    tgt_seqs, tgt_lens = make_minibatch(tgt_seqs)
    
    return src_seqs, src_lens, tgt_seqs, tgt_lens


def build_data_loader(src_seqs, tgt_seqs, vocab, batch_size=32):
    """
    Build a custom data loader for polynomial expansion
    Args:
        src: tuple of factorized inputs
        tgt: tuple of expanded target
    Returns:
        data_loader: data loader for polynomial expansion 
    """
    # Build a custom dataset
    dataset = PolyExpDataset(src_seqs, tgt_seqs, vocab)
    
    # Build a custom data loader based on the dataset
    # It returns (src_seqs, src_lens, tgt_seqs, tgt_lens) for each iteration
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              collate_fn=collate_fn)

    return data_loader


def split_train_valid_test(data, train_ratio=0.8):
    random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    valid_test_size = (len(data) - train_size) // 2
    train_split = data[:train_size]
    valid_split = data[train_size:train_size + valid_test_size]
    test_split = data[train_size + valid_test_size:]
    return train_split, valid_split, test_split
