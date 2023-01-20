import torch.nn as nn
from .baseRNN import BaseRNN


class EncoderRNN(BaseRNN):
    """
    Apply RNN on the factorized input
    Args:
        vocab_size (int): size of vocabulary
        max_len (int): maximum length allowed to be processed
        hidden_size (int): dims in the hidden state
        input_dropout (float, optional): dropout probability for the input sequence (default: 0)
        dropout (float, optional): dropout probability for the input sequence (default: 0)
        n_layer (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): whether to additionally conduct encoding in reversed order
        rnn_cell (str, optional): type of RNN cell (default: gru)
    
    Inputs: inputs, input_lens
        - **inputs**: a mini batch of token ID sequences in size (batch_size, max_len_of_seq)
        - **input_lens**: the lengths of inputs in the mini batch 
    
    Outputs: output, hidden
        -**output**: tensor containing the encoded features of inputs (batch_size, seq_len, hidden_size)
        -**hidden**: (num_layers * num_directions, batch_size, hidden_size): tnesor containing the features in hidden states
    
    Example::
        >>> encoder = EncoderRNN(vocab_size, hidden_size)
        >>> output, hidden = encoder(inputs)
    """
    def __init__(self, vocab_size, max_len, hidden_size, input_dropout_p=0, 
                 dropout_p=0, n_layer=1, bidirectional=False, rnn_cell='gru'):
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size, 
                                         input_dropout_p, dropout_p, n_layer, rnn_cell)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layer, batch_first=True, 
                                 bidirectional=bidirectional, dropout=dropout_p)


    def forward(self, inputs, input_lens=None):
        embedded = self.embedding(inputs)
        embedded = self.input_dropout(embedded)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lens, batch_first=True)
        output, hidden = self.rnn(embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden


        
