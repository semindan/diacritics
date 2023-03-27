import torch
import torch.nn as nn
from eval import evaluate_data
from utils import word_to_onehot, word_to_indexes

def train(model, data, dev, epochs=3, learning_rate=0.001, verbose = True):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        if verbose: print("epoch:", epoch)
        for idx in range(len(data["reference"])):
            hidden = model.init_hidden()

            example = word_to_onehot(data["stripped"][idx])
            reference = word_to_indexes(data["reference"][idx])

            loss = 0
            optimizer.zero_grad()

            for i in range(example.size()[0]):
                output, hidden = model(example[i], hidden)
                loss += loss_fn(output, reference[i].unsqueeze(dim=0))

            loss.backward()
            optimizer.step()

        if verbose:
            results, _ = evaluate_data(dev["stripped"], dev["reference"], model = model)
            print("dev:", results)
