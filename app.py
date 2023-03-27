import torch
import pickle
import sys
from model import RNN
from train import train
from eval import evaluate_data, predict_data_model_and_map
from utils import map_words, vocabulary

def main(text):
    model = RNN(len(vocabulary), 128, len(vocabulary))
    model.load_state_dict(torch.load("model.pt"))
    word_mapping = pickle.load(open("word_mapping.pkl", "rb"))    

    model.eval()
    prediction = predict_data_model_and_map(text.split("\n"), model, word_mapping)
    print(prediction)

if __name__ == "__main__":
    text = sys.stdin.read()
    main(text)
