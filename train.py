from os.path import exists

from CONST import RAW_DATA_DIR, TRAIN_DIR, VALID_DIR, TEST_DIR, MAX_SEQUENCE_LENGTH, TOKENS
from corpus_utils import Vocab, build_data_loader, split_train_valid_test, read_txt, write_txt, load_file
from model_utils import Fac2Exp
from trainer_utils import Perplexity, SupervisedTrainer

import torch


# Configure models
max_len = 33
hidden_size = 200
input_dropout_p = 0.2
dropout_p = 0.2
n_layer = 1
bidirectional = False
rnn_cell = 'gru'
use_attention = True


# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
batch_size = 50
n_epochs = 10
epoch = 0
print_every = 100
evaluate_every = 100
checkpoint_every = 2000
resume = False
expt_dir = "experiment/"



def run():
    if resume is False:
        print("Building vocabularies...")
        vocab = Vocab()
        vocab.build_vocab(TOKENS)
        print("Vocabulary size:", vocab.tokens_size)

        if (exists(TRAIN_DIR) and exists(VALID_DIR) and exists(TEST_DIR)) is False: 
            print("Separating train, validation, test set...")
            trainset, validset, testset = split_train_valid_test(read_txt(RAW_DATA_DIR))
            print("Dumping train, validation, test sets...")
            write_txt(TRAIN_DIR, trainset)
            write_txt(VALID_DIR, validset)
            write_txt(TEST_DIR, testset)
    
        print("Loading train, valid sets...")
        train_srcs, train_tgts = load_file(TRAIN_DIR)
        assert len(train_srcs) == len(train_tgts), \
        "Input and Output should have the same number of lines!"
        print("Train set Size:", len(train_srcs))
        train_loader = build_data_loader(train_srcs, train_tgts, vocab)
    
        print("Loading validation set...")
        valid_srcs, valid_tgts = load_file(VALID_DIR)
        assert len(valid_srcs) == len(valid_tgts), \
        "Input and Output should have the same number of lines!"
        print("Validation set Size:", len(valid_srcs))
        valid_loader = build_data_loader(valid_srcs, valid_tgts, vocab)

        print("Building model...")
        model = Fac2Exp(vocab.tokens_size, max_len, hidden_size, 
                    vocab.SOS, vocab.EOS, input_dropout_p=input_dropout_p, 
                    dropout_p=dropout_p, n_layer=n_layer, 
                    bidirectional=bidirectional, rnn_cell=rnn_cell, 
                    use_attention=use_attention)
    
        print("Set up trainer...")
        # prepare loss
        weight = torch.ones(vocab.tokens_size)
        pad = vocab.PAD
        loss = Perplexity(weight, pad)
        if torch.cuda.is_available():
            loss.cuda()

        optimizer = None
        if torch.cuda.is_available():
            model.cuda()

        for param in model.parameters():
            param.data.uniform_(-0.08, 0.08)
       
        print("Start training...")
        t = SupervisedTrainer(vocab, loss=loss, batch_size=batch_size,
                              checkpoint_every=checkpoint_every,
                              print_every=print_every, expt_dir=expt_dir) 
        model = t.train(model, train_loader, n_epochs=n_epochs, 
                        valid_loader=valid_loader, optimizer=optimizer, 
                        teacher_forcing_ratio=teacher_forcing_ratio, resume=resume)
    



if __name__ == "__main__":
    run()



