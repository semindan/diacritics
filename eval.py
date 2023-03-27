import torch
from utils import (
    accuracy,
    my_unidecode,
    word_to_onehot,
    index_to_letter,
    preprocess_word,
)

def evaluate_data(data, data_ref, model=None, word_mapping=None):
    results = {}
    predictions = []
    data_ref_text = "\n".join(data_ref)

    baseline_prediction = predict_baseline(data)
    assert len(baseline_prediction) == len(data_ref_text)
    results["baseline"] = accuracy(baseline_prediction, data_ref_text)

    if model:
        model_only_prediction = predict_data_model_only(data, model)
        assert len(model_only_prediction) == len(data_ref_text)
        results["model_only"] = accuracy(model_only_prediction, data_ref_text)
        predictions.append(model_only_prediction)

    if word_mapping:
        mapping_only_prediction = predict_data_mapping_only(data, word_mapping)
        assert len(mapping_only_prediction) == len(data_ref_text)
        results["mapping_only"] = accuracy(mapping_only_prediction, data_ref_text)
        predictions.append(mapping_only_prediction)

    if model and word_mapping:
        model_and_mapping_prediction = predict_data_model_and_map(
            data, model, word_mapping
        )
        assert len(model_and_mapping_prediction) == len(data_ref_text)
        results["model_and_mapping"] = accuracy(
            model_and_mapping_prediction, data_ref_text
        )
        predictions.append(model_and_mapping_prediction)

    return results, predictions


def predict_baseline(data):
    predictions = []
    for sequence in data:
        prediction = my_unidecode(sequence)
        predictions.append(prediction)
    return "\n".join(predictions)


def predict_data_mapping_only(data, word_mapping):
    prediction = []
    for sequence in data:
        predicted_text = []
        for word in sequence.split(" "):
            if not word.isalpha():
                predicted_text.append(word)
                continue

            preprocessed_word = preprocess_word(word)
            assert len(preprocessed_word) == len(word)
            predicted_word = word
            if preprocessed_word in word_mapping:
                predicted_word = mapping_predict_word(word_mapping, preprocessed_word) 

            predicted_word = (
                predicted_word.title() if word.istitle() else predicted_word
            )
            predicted_text.append(predicted_word)

        predicted_text = " ".join(predicted_text)
        prediction.append(predicted_text)
    return "\n".join(prediction)


def predict_data_model_only(data, model):
    predictions_model = []

    for idx, sequence in enumerate(data):
        if len(sequence) == 0:
            predictions_model.append("")
            continue

        pred = model_predict_sequence(sequence, model)
        assert len(pred) == len(sequence)
        predictions_model.append(pred)

    predicted_text_model = "\n".join(predictions_model)
    return predicted_text_model


def predict_data_model_and_map(data, model, word_mapping):
    predictions = []

    for idx, sequence in enumerate(data):
        predicted_text = []
        for word in sequence.split(" "):
            if not word.isalpha():
                predicted_text.append(word)
                continue

            preprocessed_word = preprocess_word(word)
            assert len(preprocessed_word) == len(word)
            predicted_word = word

            if preprocessed_word in word_mapping:
                predicted_word = mapping_predict_word(word_mapping, preprocessed_word)
            else:
                predicted_word = model_predict_word(model, preprocessed_word)
                assert len(predicted_word) == len(word)

            predicted_word = (
                predicted_word.title() if word.istitle() else predicted_word
            )
            predicted_text.append(predicted_word)

        predicted_text = " ".join(predicted_text)
        predictions.append(predicted_text)

    return "\n".join(predictions)


def mapping_predict_word(word_mapping, preprocessed_word):
    return max(word_mapping[preprocessed_word].items(), key=lambda x: x[1])[0]

def model_predict_sequence(text, model):
    predicted_text = []
    text_split = text.split(" ")
    for word in text_split:
        if not word.isalpha():
            predicted_text.append(word)
            continue

        predicted_word = model_predict_word(model, word)
        predicted_word = predicted_word.title() if word.istitle() else predicted_word
        predicted_text.append(predicted_word)

    predicted_text = " ".join(predicted_text)
    return predicted_text


def model_predict_word(model, word):
    hidden = model.init_hidden()
    preprocessed_word = preprocess_word(word)
    word_tensor = word_to_onehot(preprocessed_word)

    predicted_word = []
    for i in range(word_tensor.size()[0]):
        output, hidden = model(word_tensor[i], hidden)
        predicted_letter = index_to_letter(int(torch.argmax(output)))
        if predicted_letter == "<unk>":
            predicted_letter = index_to_letter(int(torch.topk(output, k=2).values[-1][1]))

        # if the model is mistaken, we fall back to the original letter we received 
        if my_unidecode(predicted_letter) != preprocessed_word[i]:
            predicted_letter = preprocessed_word[i]

        predicted_word.append(predicted_letter)

    predicted_word = "".join(predicted_word)
    return predicted_word
