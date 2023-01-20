# Datasets
DATA_ROOT = "corpus/"
RAW_DATA_DIR = DATA_ROOT + "train.txt"
TRAIN_DIR = DATA_ROOT + "train_split.txt"
VALID_DIR = DATA_ROOT + "valid_split.txt"
TEST_DIR = DATA_ROOT + "test_split.txt"

# Build vocabulary
MAX_SEQUENCE_LENGTH = 29
TOKENS = list("0123456789abcdefghijklmnopqrstuvwxyz()+-*/")
TOKENS += ['tan', 'sin', 'cos', \
           'tanh', 'sinh', 'cosh', \
           'coth', 'sech', 'sech', 'log']
