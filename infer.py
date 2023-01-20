from trainer_utils import Checkpoint
from corpus_utils import load_file
from eval_utils import Inference


checkpoint_path = "experiment/checkpoints/2022_08_04_15_23_20"
test_dir = "corpus/test_split.txt"



def run_inference(f_dir):
    checkpoint = Checkpoint.load(checkpoint_path)
    fac2exp = checkpoint.model
    fac2exp.infer = True
    vocab = checkpoint.vocab
    infer_oper = Inference(fac2exp, vocab)

    src, tgt = load_file(f_dir)
    
    count = 0
    correct = 0
    for i, (s, t) in enumerate(zip(src, tgt)):
        pred = infer_oper.predict(s)[:-1]
        pred = "".join(pred)
        count += 1
        if pred == t:
            correct += 1
            print("i=", i)
            print("s=", s)
            print("predict=", "".join(pred))
            print("gold=", t)
            print("---------------")
    print(count, correct, correct / count)

if __name__ == "__main__":
    run_inference(test_dir)
