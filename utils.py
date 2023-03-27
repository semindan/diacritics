import string
from unidecode import unidecode
import torch

vocabulary = [
    "<unk>",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "á",
    "é",
    "í",
    "ó",
    "ú",
    "ý",
    "č",
    "ď",
    "ě",
    "ľ",
    "ň",
    "ř",
    "š",
    "ť",
    "ů",
    "ž",
]


def preprocess_word(word):
    word = word.lower()
    remove_punctuation = str.maketrans(
        string.punctuation, " " * len(string.punctuation)
    )
    word = word.translate(remove_punctuation).replace(" ", "")
    return word

def accuracy(predicted_text, reference_text):
    correct = 0
    total = 0
    for prediction, reference in zip(list(predicted_text), list(reference_text)):
        if reference != " ":
            correct += 1 if prediction == reference else 0
            total += 1
    return correct / total

def my_unidecode(text):
    new_text = ""
    for char in text:
        if len(unidecode(char)) != len(char):
            new_text += "?"
        else:
            new_text += unidecode(char)
    return new_text

def map_words(data, data_ref):
    word_mapping = {}
    for word, ref_word in zip(data, data_ref):
        if word not in word_mapping:
            word_mapping[word] = {}
        word_mapping[word][ref_word] = word_mapping[word].get(ref_word, 0) + 1
    return word_mapping

def index_to_letter(index):
    return vocabulary[index]

def letter_to_index(letter):
    if letter not in vocabulary:
        letter = "<unk>"
    return vocabulary.index(letter)

def letter_to_onehot(letter):
    tensor = torch.zeros(1, len(vocabulary))
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def word_to_onehot(word):
    tensor = torch.zeros(len(word), 1, len(vocabulary))
    for index, letter in enumerate(word):
        tensor[index][0][letter_to_index(letter)] = 1
    return tensor

def word_to_indexes(word):
    tensor = torch.zeros(len(word)).long()
    for index, letter in enumerate(word):
        tensor[index] = letter_to_index(letter)
    return tensor