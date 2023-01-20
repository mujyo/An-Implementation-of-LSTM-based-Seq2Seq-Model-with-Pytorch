import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Apply attention to select most relevant src to the current decoder hidden state
    x = context * 
    Args:
        dim(int): The dim number of hidden state
    
    Inputs: decoder_hidden, encoder_hidden
        - **decoder_hidden** (batch_size, output_len, hidden_size): decoder's hidden state
        - **encoder_hidden** (batch_size, input_len, hidden_size): encoder's hidden state
    
    Outputs: output, attn
        - **output** (batch_size, output_len, dimensions): decoder's hidden state created using attention weights
        - **attn** (batch_size, output_len, input_len): attention weights
    
    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::
        >>> attention = seq2seq.models.Attention(256)
        >>> context = Variable(torch.randn(5, 3, 256))
        >>> output = Variable(torch.randn(5, 5, 256))
        >>> output, attn = attention(output, context)
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None


    def set_mask(self, mask):
        """
        Set indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, decoder_hidden, encoder_hidden):
        batch_size = decoder_hidden.size(0)
        hidden_size = decoder_hidden.size(2)
        input_size = encoder_hidden.size(1)
        # (batch_size, out_len, dim) * (batch_size, in_len, dim) -> (batch_size, out_len, in_len)
        attn = torch.bmm(decoder_hidden, encoder_hidden.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, encoder_hidden)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, decoder_hidden), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn

