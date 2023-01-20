from __future__ import print_function, division

import torch
import torchtext

from .loss import NLLLoss


class Evaluator(object):
    """ Class to evaluate models with given datasets.
    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, batch_iterator, vocab):
        """ Evaluate a model on given dataset and return performance.
        Args:
            model (seq2seq.models): model to evaluate
            batch_iterator (seq2seq.dataset.dataset.Dataset): dataset to evaluate against
            vocab: the vocabulary recording token_index alignments
        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()
        model.infer = True

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        pad = vocab.PAD

        with torch.no_grad():
            for batch in batch_iterator:
                src_seqs, src_lens, tgt_seqs, tgt_lens = batch
                decoder_outputs, decoder_hidden, other = model(src_seqs, src_lens, tgt_seqs)

                # Evaluation
                seqlist = other['sequence']
                for step, step_output in enumerate(decoder_outputs):
                    target = tgt_seqs[:, step + 1]
                    loss.eval_batch(step_output.view(tgt_seqs.size(0), -1), target)

                    non_padding = target.ne(pad)
                    correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                    match += correct
                    total += non_padding.sum().item()

        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy
