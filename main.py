import argparse
import torch
import pickle
from model import RNN
from data import CzTextCorpus
from train import train
from eval import evaluate_data
from utils import map_words, vocabulary

def main(args):
    word_mapping = {}
    model = RNN(len(vocabulary), 128, len(vocabulary))
    if args.use_pretrained:
        data = CzTextCorpus(train_size=args.train_size, path_train=None, path_dev=None, path_test=args.path_test)
        model.load_state_dict(torch.load("model.pt"))
        word_mapping = pickle.load(open("word_mapping.pkl", "rb"))
    else:
        data = CzTextCorpus(train_size=args.train_size, path_train=args.path_train, path_dev=args.path_dev, path_test=args.path_test)
        word_mapping = map_words(data.words["stripped"], data.words["reference"])
        train(model, data.train_words, data.dev, epochs=3, verbose=args.verbose)

    model.eval()
    results, predictions = evaluate_data(
        data.test["stripped"],
        data.test["reference"],
        model=model,
        word_mapping=word_mapping,
    )

    if args.verbose:
        print("results on test:", results)
    if not args.no_prediction_print:
        print(predictions[-1])

    if not args.use_pretrained:
        torch.save(model.state_dict(), "model.pt")
        pickle.dump(word_mapping, open("word_mapping.pkl", "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, help="train_size", default=30000)
    parser.add_argument("--path_train", type=str, help="path_train", default=None)
    parser.add_argument("--path_dev", type=str, help="path_dev", default=None)
    parser.add_argument("--path_test", type=str, help="path_test", default=None)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no_prediction_print", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_pretrained", action=argparse.BooleanOptionalAction, default=False)
    
    args = parser.parse_args()
    main(args)

