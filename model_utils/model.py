import torch.nn as nn
import torch.nn.functional as F

from .encoder import EncoderRNN 
from .decoder import DecoderRNN

"""
Implementation refers to https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/seq2seq.py
"""

class Fac2Exp(nn.Module):
    """
    Standard seq2seq architecture
    Args: 
        vocab_num:
        hidden_size:
        n_layers:
        attn_model:
        dropout:

    """
    def __init__(self, vocab_num, max_len, hidden_size, sos_id, eos_id, input_dropout_p=0, 
                 dropout_p=0, n_layer=1, bidirectional=False, rnn_cell='gru', use_attention=False, infer=False):
        super(Fac2Exp, self).__init__()
        self.encoder = EncoderRNN(vocab_num, max_len, hidden_size, 
                                  input_dropout_p=input_dropout_p, dropout_p=dropout_p, 
                                  n_layer=n_layer, bidirectional=bidirectional, rnn_cell=rnn_cell)
        self.decoder = DecoderRNN(self.encoder.embedding, vocab_num, max_len, hidden_size,
                                  sos_id, eos_id, n_layer=n_layer, rnn_cell=rnn_cell, 
                                  bidirectional=bidirectional, input_dropout_p=input_dropout_p, 
                                  dropout_p=dropout_p, use_attention=use_attention)
        self.decode_func = F.log_softmax
        self.infer = infer


    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_var, input_len=None, target_var=None,
                teacher_forcing_ratio=0):
        if self.infer:
            teacher_forcing_ratio = 0
        encoder_outputs, encoder_hidden = self.encoder(input_var, input_len)
        output = self.decoder(inputs=target_var, 
                              encoder_hidden=encoder_hidden, 
                              encoder_outputs=encoder_outputs, 
                              function=self.decode_func, 
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return output



