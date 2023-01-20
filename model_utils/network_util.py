import torch
import torch.nn as nn

def init_decoder_state(encoder_hidden):
    """
    Initialize decoder's first hidden state using the last hidden state
    of encoder
    Args:
        hidden: the last hidden state of encoder
    
    Return:
        decoder_initial: can be fed into the decoder at the firt step
    """
    if encoder_hidden is None:
        return None

    # If we use bidirectional encoder, we concate the two vectors for both directions
    if isinstance(encoder_hidden, tuple):
        decoder_initial = tuple([cat_directions(h) for h in encoder_hidden])
    else:
        #decoder_initial = cat_directions(encoder_hidden)
        decoder_initial = encoder_hidden
    return decoder_initial


def cat_directions(hidden):
    """
    To deal with hidden states coming from bidirectional encoder
    (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
    Args:
        hidden: a hidden state from bidirectional encoder
    
    Return:
        concated_hidden: the concated hidden state which serves as the input to decoder
    """
    concated_hidden = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)
    return concated_hidden



def validate_args(inputs, encoder_hidden, encoder_outputs, function, 
                  teacher_forcing_ratio, use_attention=False, 
                  rnn_cell=nn.GRU, sos_id=None, max_len=None):
    """
    Infer the size of mini-batch during decode
    """
    if use_attention:
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

    # inference batch size
    if inputs is None and encoder_hidden is None:
        batch_size = 1
    else:
        if inputs is not None:
            batch_size = inputs.size(0)
        else:
            if rnn_cell is nn.LSTM:
                batch_size = encoder_hidden[0].size(1)
            elif rnn_cell is nn.GRU:
                batch_size = encoder_hidden.size(1)

    # set default input and max decoding length
    if inputs is None:
        teacher_forcing_ratio = 0
        #if teacher_forcing_ratio > 0:
        #    raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
        inputs = torch.LongTensor([sos_id] * batch_size).view(batch_size, 1)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
    else:
        max_len = inputs.size(1) - 1 # minus the start of sequence symbol

    return inputs, batch_size, max_len
