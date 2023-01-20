import torch
from torch.autograd import Variable


class Inference(object):
    """
    Predict the expansion given factorized string
    Args:
        model (Fac2Exp model): trained model. To load a model, use checkpoint.load
        vocab: token_ID alignments for both factorized input domain and expanded output domain
    """
    def __init__(self, model, vocab):
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.model.teacher_forcing_ratio = 0
        self.vocab = vocab

    def expand_polynomial(self, src_seq):
        src_ids = self.vocab.transform_to_index(src_seq) + [self.vocab.EOS]
        src_len = len(src_ids)
        src_ids = torch.LongTensor(src_ids).view(1, -1)
        if torch.cuda.is_available():
            src_ids = src_ids.cuda()
        with torch.no_grad():
            _, _, memory = self.model(src_ids, [src_len])
        return memory

    def predict(self, src_seq):
        """
        Predict the expanded form of factorized polynomial
        Args:
            src_seq (string): factorized form

        Returns:
            tgt_seq (list): list of tokens in expanded polynomial
        """
        memory = self.expand_polynomial(src_seq)
        length = memory['length'][0]
        
        expanded_ids = [memory['sequence'][i][0].data[0] for i in range(length)]
        tgt_seq = [self.vocab.indx2tok[indx.item()] for indx in expanded_ids]
        
        return tgt_seq
        
