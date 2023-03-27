from utils import my_unidecode, preprocess_word

class CzTextCorpus:
    def __init__(self, path_train="text.txt", path_dev="dev.txt", path_test="test.txt", train_size = None):
        self.test = self.prepare_data(self.load_data(path_test))

        if path_dev:
            self.dev = self.prepare_data(self.load_data(path_dev))
        
        if path_train:
            train = self.load_data(path_train)
            words = self.prepare_words(train)
            self.words = self.prepare_data(words)

            train_words = self.prepare_train_words(train)
            train_words = train_words[:train_size] if train_size else train_words
            self.train_words = self.prepare_data(train_words)

    def load_data(self, path):
        data = []
        with open(path, "r") as dump_f:
            text = dump_f.read()
            data += text.split("\n")
        return data

    def prepare_train_words(self, data):
        words = self.prepare_words(data)
        words = list(filter(lambda word: len(word) > 2, words))
        return words

    def prepare_words(self, data):
        words = " ".join(data)
        words = words.split()
        words = map(lambda word: preprocess_word(word), words)
        words = list(filter(lambda word: word.isalpha(), words))
        return words

    def prepare_data(self, data):
        return {
            "stripped": [my_unidecode(data_i) for data_i in data],
            "reference": [data_i for data_i in data],
        }
