import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import blitz
from tqdm import tqdm
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from blitz.losses import kl_divergence_from_nn

# --- HP ---
BATCH_SIZE = 200
EPOCHS = 60
kl_div_values = []
kl_div_per_param_values = []


# --- utils ---

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_convergence_over_epochs(train_list: list, test_list: list, epochs: int, mode: str, model: int) -> None:
    plt.plot(range(1, epochs + 1), train_list)
    plt.plot(range(1, epochs + 1), test_list)
    plt.xlabel('Epochs')
    plt.ylabel(f'{mode}')
    plt.title(f"Model {model}'s {mode} over epochs")
    plt.legend(['Train', 'Test'])
    plt.show()


def get_MNIST(Limit=None, ber_lables=False):
    """
    Fetching PyTorch's MNIST dataset.
    taking some of the digits is optional
    :param Limit: the pool of digits to fetch
    :return: train,test,dataset,loader
    """
    if Limit is None:
        Limit = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

    # limiting the training data
    filter_indices = list(np.argwhere(np.isin(train_dataset.targets, Limit)).ravel())
    train_dataset.data = train_dataset.data[filter_indices, :, :]
    train_dataset.targets = train_dataset.targets[filter_indices]

    test_dataset = datasets.MNIST(root='./data/',
                                  train=False,
                                  transform=transform,
                                  download=True)
    # limiting the Test data
    filter_indices = list(np.argwhere(np.isin(test_dataset.targets, Limit)).ravel())
    test_dataset.data = test_dataset.data[filter_indices, :, :]
    test_dataset.targets = test_dataset.targets[filter_indices]

    if ber_lables:
        test_dataset.targets = torch.bernoulli(torch.full((len(test_dataset.targets),), 0.5))
        train_dataset.targets = torch.bernoulli(torch.full((len(train_dataset.targets),), 0.5))

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)

    return train_dataset, train_loader, test_dataset, test_loader


class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.blinear1 = BayesianLinear(input_size, 256)
        self.blinear2 = BayesianLinear(256, output_size)

    def forward(self, x):
        x_ = self.blinear1(x)
        x_ = F.relu(x_)
        return self.blinear2(x_)


class BayesianNeuralNetwork2(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.blinear1 = BayesianLinear(input_size, 512)
        self.blinear2 = BayesianLinear(512, output_size)

    def forward(self, x):
        x_ = self.blinear1(x)
        x_ = F.tanh(x_)
        return self.blinear2(x_)


class BayesianNeuralNetwork3(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.blinear1 = BayesianLinear(input_size, 512)
        self.blinear2 = BayesianLinear(512, 256)
        self.blinear3 = BayesianLinear(256, output_size)

    def forward(self, x):
        x_ = self.blinear1(x)
        x_ = F.relu(x_)
        x_ = self.blinear2(x_)
        x_ = F.tanh(x_)
        return self.blinear3(x_)


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


def train_model1():
    train_data, train_loader, test_data, test_loader = get_MNIST()

    print("Model 1 - No randomization, all MNIST dataset:")
    model_1 = BayesianNeuralNetwork(input_size=28 * 28,
                                    output_size=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_1.parameters())

    train_accuracies_1, train_losses_1, test_accuracies_1, test_losses_1 = [], [], [], []
    for i in range(EPOCHS):  # Running EPOCH times over the entire dataset
        agg_train_loss, agg_train_correct, agg_test_loss, agg_test_correct = 0, 0, 0, 0

        print(f'Epoch {i + 1}')
        epoch_train_loss, epoch_test_loss = 0, 0
        # Training phase
        model_1.train(True)
        for (train_images, train_labels) in train_loader:
            train_images = train_images.view(-1, 28 * 28)
            optimizer.zero_grad()
            train_outputs = model_1(train_images)
            loss = criterion(train_outputs, train_labels.long())
            loss.backward()
            optimizer.step()
            # Calculating and accumulating loss
            agg_train_loss += loss.item()
            _, predicted = torch.max(train_outputs, dim=1)
            agg_train_correct += (predicted == train_labels).sum()
        train_losses_1.append(agg_train_loss / len(train_loader))
        train_accuracies_1.append(agg_train_correct / len(train_data))
        with torch.no_grad():
            for (test_images, test_labels) in test_loader:
                test_images = test_images.view(-1, 28 * 28)
                test_outputs = model_1(test_images)
                test_predictions = torch.argmax(test_outputs, dim=1)
                loss = criterion(test_outputs, test_labels)
                agg_test_loss += loss.item()
                _, predicted = torch.max(test_outputs, dim=1)
                agg_test_correct += (test_predictions == test_labels).sum()
            test_losses_1.append(agg_test_loss / len(test_loader))
            test_accuracies_1.append(agg_test_correct / len(test_data))

    # Plotting accuracy and loss graphs
    plot_convergence_over_epochs(train_accuracies_1, test_accuracies_1, epochs=EPOCHS, mode='Accuracy', model=1)
    # plot_convergence_over_epochs(train_losses_1, test_losses_1, epochs=EPOCHS, mode='CE Loss', model=1)
    # KL
    kl_div_value = kl_divergence_from_nn(model=model_1)
    kl_div_values.append(kl_div_value)
    kl_div_per_param_values.append(kl_div_value / count_parameters(model_1))
    # print(f"Model {1}'s KL Divergence: {kl_div_values[0]}; Per parameter: {kl_div_per_param_values[0]}")
    torch.save(model_1, 'model1.pkl')
    return {"model": model_1, "kl_val": kl_div_value, "train_losses": train_losses_1,
            "train_accuracies": train_accuracies_1}


def train_model2():
    train_data, train_loader, test_data, test_loader = get_MNIST()

    print("Model 2 - No randomization, first 200 from MNIST dataset:")
    model_2 = BayesianNeuralNetwork(input_size=28 * 28,
                                    output_size=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_2.parameters())
    train_images, train_labels = next(iter(train_loader))  # First 200
    train_images = train_images.view(-1, 28 * 28)

    train_accuracies_2, train_losses_2, test_losses_2, test_accuracies_2 = [], [], [], []
    for i in range(EPOCHS):  # Running EPOCH times over the entire dataset
        agg_train_loss, agg_train_correct, agg_test_loss, agg_test_correct = 0, 0, 0, 0

        print(f'Epoch {i + 1}')
        epoch_train_loss, epoch_test_loss = 0, 0
        # Training phase
        model_2.train(True)
        train_images = train_images.view(-1, 28 * 28)
        optimizer.zero_grad()
        train_outputs = model_2(train_images)
        loss = criterion(train_outputs, train_labels.long())
        loss.backward()
        optimizer.step()
        # Calculating and accumulating loss
        agg_train_loss += loss.item()
        _, predicted = torch.max(train_outputs, dim=1)
        agg_train_correct += (predicted == train_labels).sum()

        train_losses_2.append(agg_train_loss)
        train_accuracies_2.append(agg_train_correct / len(train_labels))
        for (test_images, test_labels) in test_loader:
            test_images = test_images.view(-1, 28 * 28)
            test_outputs = model_2(test_images)
            test_predictions = torch.argmax(test_outputs, dim=1)
            loss = criterion(test_outputs, test_labels)
            agg_test_loss += loss.item()
            _, predicted = torch.max(test_outputs, dim=1)
            agg_test_correct += (test_predictions == test_labels).sum()
        test_losses_2.append(agg_test_loss / len(test_loader))
        test_accuracies_2.append(agg_test_correct / len(test_data))

    # Plotting accuracy and loss graphs
    plot_convergence_over_epochs(train_accuracies_2, test_accuracies_2, epochs=EPOCHS, mode='Accuracy', model=2)
    # plot_convergence_over_epochs(train_losses_2, test_losses_2, epochs=EPOCHS, mode='CE Loss', model=1)

    # KL
    kl_div_value = kl_divergence_from_nn(model=model_2)
    kl_div_values.append(kl_div_value)
    kl_div_per_param_values.append(kl_div_value / count_parameters(model_2))
    # print(f"Model {2}'s KL Divergence: {kl_div_values[0]}; Per parameter: {kl_div_per_param_values[0]}")
    torch.save(model_2, 'model2.pkl')
    return {"model": model_2, "kl_val": kl_div_value, "train_losses": train_losses_2,
            "train_accuracies": train_accuracies_2}


def train_model3():
    train_data, train_loader, test_data, test_loader = get_MNIST([3, 8])

    print("Model 3 - No randomization, trained on the 200 first 3's and 8's:")
    model_3 = BayesianNeuralNetwork(input_size=28 * 28,
                                    output_size=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_3.parameters())
    train_images, train_labels = next(iter(train_loader))  # First 200
    train_images = train_images.view(-1, 28 * 28)

    train_accuracies_3, train_losses_3, test_losses, test_accuracies = [], [], [], []
    for i in range(EPOCHS):  # Running EPOCH times over the entire dataset
        agg_train_loss, agg_train_correct, agg_test_loss, agg_test_correct = 0, 0, 0, 0
        print(f'Epoch {i + 1}')
        epoch_train_loss, epoch_test_loss = 0, 0
        # Training phase
        model_3.train(True)
        train_images = train_images.view(-1, 28 * 28)
        optimizer.zero_grad()
        train_outputs = model_3(train_images)
        loss = criterion(train_outputs, train_labels.long())
        loss.backward()
        optimizer.step()
        # Calculating and accumulating loss
        agg_train_loss += loss.item()
        _, predicted = torch.max(train_outputs, dim=1)
        agg_train_correct += (predicted == train_labels).sum()

        train_losses_3.append(agg_train_loss)
        train_accuracies_3.append(agg_train_correct / len(train_labels))
        for (test_images, test_labels) in test_loader:
            test_images = test_images.view(-1, 28 * 28)
            test_outputs = model_3(test_images)
            test_predictions = torch.argmax(test_outputs, dim=1)
            loss = criterion(test_outputs, test_labels)
            agg_test_loss += loss.item()
            _, predicted = torch.max(test_outputs, dim=1)
            agg_test_correct += (test_predictions == test_labels).sum()
        test_losses.append(agg_test_loss / len(test_loader))
        test_accuracies.append(agg_test_correct / len(test_data))

    # Plotting accuracy and loss graphs
    plot_convergence_over_epochs(train_accuracies_3, test_accuracies, epochs=EPOCHS, mode='Accuracy', model=3)
    # plot_convergence_over_epochs(train_losses_3, test_losses, epochs=EPOCHS, mode='CE Loss', model=1)

    # KL
    kl_div_value = kl_divergence_from_nn(model=model_3)
    kl_div_values.append(kl_div_value)
    kl_div_per_param_values.append(kl_div_value / count_parameters(model_3))
    # print(f"Model {3}'s KL Divergence: {kl_div_values[0]}; Per parameter: {kl_div_per_param_values[0]}")
    torch.save(model_3, 'model3.pkl')
    return {"model": model_3, "kl_val": kl_div_value, "train_losses": train_losses_3,
            "train_accuracies": train_accuracies_3}


def train_model4():
    train_data, train_loader, test_data, test_loader = get_MNIST()

    print("Model 4 - No randomization, trained on all 3's and 8's ")
    model_4 = BayesianNeuralNetwork(input_size=28 * 28,
                                    output_size=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_4.parameters())

    train_accuracies_4, train_losses_4, test_losses, test_accuracies = [], [], [], []
    for i in range(EPOCHS):  # Running EPOCH times over the entire dataset
        agg_train_loss, agg_train_correct, agg_test_loss, agg_test_correct = 0, 0, 0, 0
        print(f'Epoch {i + 1}')
        epoch_train_loss, epoch_test_loss = 0, 0
        # Training phase
        model_4.train(True)
        for (train_images, train_labels) in train_loader:
            train_images = train_images.view(-1, 28 * 28)
            optimizer.zero_grad()
            train_outputs = model_4(train_images)
            loss = criterion(train_outputs, train_labels.long())
            loss.backward()
            optimizer.step()
            # Calculating and accumulating loss
            agg_train_loss += loss.item()
            _, predicted = torch.max(train_outputs, dim=1)
            agg_train_correct += (predicted == train_labels).sum()

        train_losses_4.append(agg_train_loss / len(train_loader))
        train_accuracies_4.append(agg_train_correct / len(train_data))
        for (test_images, test_labels) in test_loader:
            test_images = test_images.view(-1, 28 * 28)
            test_outputs = model_4(test_images)
            test_predictions = torch.argmax(test_outputs, dim=1)
            loss = criterion(test_outputs, test_labels)
            agg_test_loss += loss.item()
            _, predicted = torch.max(test_outputs, dim=1)
            agg_test_correct += (test_predictions == test_labels).sum()
        test_losses.append(agg_test_loss / len(test_loader))
        test_accuracies.append(agg_test_correct / len(test_data))

    # Plotting accuracy and loss graphs
    plot_convergence_over_epochs(train_accuracies_4, test_accuracies, epochs=EPOCHS, mode='Accuracy', model=4)
    # plot_convergence_over_epochs(train_losses_4, test_losses, epochs=EPOCHS, mode='CE Loss', model=1)

    # KL
    kl_div_value = kl_divergence_from_nn(model=model_4)
    kl_div_values.append(kl_div_value)
    kl_div_per_param_values.append(kl_div_value / count_parameters(model_4))
    # print(f"Model {4}'s KL Divergence: {kl_div_values[0]}; Per parameter: {kl_div_per_param_values[0]}")
    torch.save(model_4, 'model4.pkl')
    return {"model": model_4, "kl_val": kl_div_value, "train_losses": train_losses_4,
            "train_accuracies": train_accuracies_4}


def train_model5():
    train_data, train_loader, test_data, test_loader = get_MNIST(ber_lables=True)

    print(" Model 5 - Random labels, trained on the first 200 MNIST examples ---")
    model_5 = BayesianNeuralNetwork(input_size=28 * 28,
                                    output_size=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_5.parameters())
    train_images, train_labels = next(iter(train_loader))  # First 200
    train_images = train_images.view(-1, 28 * 28)
    train_accuracies_5, train_losses_5, test_losses, test_accuracies = [], [], [], []
    for i in range(EPOCHS):  # Running EPOCH times over the entire dataset
        agg_train_loss, agg_train_correct, agg_test_loss, agg_test_correct = 0, 0, 0, 0
        print(f'Epoch {i + 1}')
        epoch_train_loss, epoch_test_loss = 0, 0
        # Training phase
        model_5.train(True)
        train_images = train_images.view(-1, 28 * 28)
        optimizer.zero_grad()
        train_outputs = model_5(train_images)
        loss = criterion(train_outputs, train_labels.long())
        loss.backward()
        optimizer.step()
        # Calculating and accumulating loss
        agg_train_loss += loss.item()
        _, predicted = torch.max(train_outputs, dim=1)
        agg_train_correct += (predicted == train_labels).sum()

        train_losses_5.append(agg_train_loss)
        train_accuracies_5.append(agg_train_correct / len(train_labels))
        for (test_images, test_labels) in test_loader:
            test_images = test_images.view(-1, 28 * 28)
            test_outputs = model_5(test_images)
            test_predictions = torch.argmax(test_outputs, dim=1)
            loss = criterion(test_outputs, test_labels)
            agg_test_loss += loss.item()
            _, predicted = torch.max(test_outputs, dim=1)
            agg_test_correct += (test_predictions == test_labels).sum()
        test_losses.append(agg_test_loss / len(test_loader))
        test_accuracies.append(agg_test_correct / len(test_data))

    # Plotting accuracy and loss graphs
    plot_convergence_over_epochs(train_accuracies_5, test_accuracies, epochs=EPOCHS, mode='Accuracy', model=5)
    # plot_convergence_over_epochs(train_losses_5, test_losses, epochs=100, mode='CE Loss', model=1)

    # KL
    kl_div_value = kl_divergence_from_nn(model=model_5)
    kl_div_values.append(kl_div_value)
    kl_div_per_param_values.append(kl_div_value / count_parameters(model_5))
    # print(f"Model {5}'s KL Divergence: {kl_div_values[0]}; Per parameter: {kl_div_per_param_values[0]}")
    torch.save(model_5, 'model5.pkl')
    return {"model": model_5, "kl_val": kl_div_value, "train_losses": train_losses_5,
            "train_accuracies": train_accuracies_5}


def train_logistic(L2_regularization=None):
    train_data, train_loader, test_data, test_loader = get_MNIST()

    print("Model logistic_regression -  all MNIST dataset:")
    model_1 = LogisticRegression(input_dim=28 * 28,
                                 output_dim=10)
    criterion = nn.CrossEntropyLoss()
    if L2_regularization == None:
        optimizer = torch.optim.Adam(model_1.parameters())
    else:
        optimizer = torch.optim.Adam(model_1.parameters(), lr=1e-4, weight_decay=1e-5)

    train_accuracies_1, train_losses_1, test_accuracies_1, test_losses_1 = [], [], [], []
    for i in range(EPOCHS):  # Running EPOCH times over the entire dataset
        agg_train_loss, agg_train_correct, agg_test_loss, agg_test_correct = 0, 0, 0, 0

        print(f'Epoch {i + 1}')
        epoch_train_loss, epoch_test_loss = 0, 0
        # Training phase
        model_1.train(True)
        for (train_images, train_labels) in train_loader:
            train_images = train_images.view(-1, 28 * 28)
            optimizer.zero_grad()
            train_outputs = model_1(train_images)
            loss = criterion(train_outputs, train_labels.long())
            loss.backward()
            optimizer.step()
            # Calculating and accumulating loss
            agg_train_loss += loss.item()
            _, predicted = torch.max(train_outputs, dim=1)
            agg_train_correct += (predicted == train_labels).sum()
        train_losses_1.append(agg_train_loss / len(train_loader))
        train_accuracies_1.append(agg_train_correct / len(train_data))
        with torch.no_grad():
            for (test_images, test_labels) in test_loader:
                test_images = test_images.view(-1, 28 * 28)
                test_outputs = model_1(test_images)
                test_predictions = torch.argmax(test_outputs, dim=1)
                loss = criterion(test_outputs, test_labels)
                agg_test_loss += loss.item()
                _, predicted = torch.max(test_outputs, dim=1)
                agg_test_correct += (test_predictions == test_labels).sum()
            test_losses_1.append(agg_test_loss / len(test_loader))
            test_accuracies_1.append(agg_test_correct / len(test_data))

    # Plotting accuracy and loss graphs

    label = "logistic_regression_with_L2_Reg" if L2_regularization else "logistic_regression_no_regularization"
    plot_convergence_over_epochs(train_accuracies_1, test_accuracies_1, epochs=EPOCHS, mode='Accuracy', model=label)
    plot_convergence_over_epochs(train_losses_1, test_losses_1, epochs=EPOCHS, mode='CE Loss', model=label)

    # KL
    kl_div_value = kl_divergence_from_nn(model=model_1)
    kl_div_values.append(kl_div_value)
    kl_div_per_param_values.append(kl_div_value / count_parameters(model_1))
    torch.save(model_1, 'model_logistic_nonreg.pkl') if L2_regularization else torch.save(model_1, 'modellogistic_l2.pkl')
    return {"model": model_1, "kl_val": kl_div_value, "train_losses": train_losses_1,
            "train_accuracies": train_accuracies_1}



def train_model1V2():
    train_data, train_loader, test_data, test_loader = get_MNIST()

    print("Model 1 - No randomization, all MNIST dataset:")
    model_1 = BayesianNeuralNetwork2(input_size=28 * 28,
                                    output_size=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_1.parameters())

    train_accuracies_1, train_losses_1, test_accuracies_1, test_losses_1 = [], [], [], []
    for i in range(EPOCHS):  # Running EPOCH times over the entire dataset
        agg_train_loss, agg_train_correct, agg_test_loss, agg_test_correct = 0, 0, 0, 0

        print(f'Epoch {i + 1}')
        epoch_train_loss, epoch_test_loss = 0, 0
        # Training phase
        model_1.train(True)
        for (train_images, train_labels) in train_loader:
            train_images = train_images.view(-1, 28 * 28)
            optimizer.zero_grad()
            train_outputs = model_1(train_images)
            loss = criterion(train_outputs, train_labels.long())
            loss.backward()
            optimizer.step()
            # Calculating and accumulating loss
            agg_train_loss += loss.item()
            _, predicted = torch.max(train_outputs, dim=1)
            agg_train_correct += (predicted == train_labels).sum()
        train_losses_1.append(agg_train_loss / len(train_loader))
        train_accuracies_1.append(agg_train_correct / len(train_data))
        with torch.no_grad():
            for (test_images, test_labels) in test_loader:
                test_images = test_images.view(-1, 28 * 28)
                test_outputs = model_1(test_images)
                test_predictions = torch.argmax(test_outputs, dim=1)
                loss = criterion(test_outputs, test_labels)
                agg_test_loss += loss.item()
                _, predicted = torch.max(test_outputs, dim=1)
                agg_test_correct += (test_predictions == test_labels).sum()
            test_losses_1.append(agg_test_loss / len(test_loader))
            test_accuracies_1.append(agg_test_correct / len(test_data))

    # Plotting accuracy and loss graphs
    plot_convergence_over_epochs(train_accuracies_1, test_accuracies_1, epochs=EPOCHS, mode='Accuracy', model="1V2")
    # plot_convergence_over_epochs(train_losses_1, test_losses_1, epochs=EPOCHS, mode='CE Loss', model=1)
    # KL
    kl_div_value = kl_divergence_from_nn(model=model_1)
    kl_div_values.append(kl_div_value)
    kl_div_per_param_values.append(kl_div_value / count_parameters(model_1))
    # print(f"Model {1}'s KL Divergence: {kl_div_values[0]}; Per parameter: {kl_div_per_param_values[0]}")
    torch.save(model_1, 'model1V2.pkl')
    return {"model": model_1, "kl_val": kl_div_value, "train_losses": train_losses_1,
            "train_accuracies": train_accuracies_1}



def train_model1V3():
    train_data, train_loader, test_data, test_loader = get_MNIST()

    print("Model 1 - No randomization, all MNIST dataset:")
    model_1 = BayesianNeuralNetwork3(input_size=28 * 28,
                                    output_size=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_1.parameters())

    train_accuracies_1, train_losses_1, test_accuracies_1, test_losses_1 = [], [], [], []
    for i in range(EPOCHS):  # Running EPOCH times over the entire dataset
        agg_train_loss, agg_train_correct, agg_test_loss, agg_test_correct = 0, 0, 0, 0

        print(f'Epoch {i + 1}')
        epoch_train_loss, epoch_test_loss = 0, 0
        # Training phase
        model_1.train(True)
        for (train_images, train_labels) in train_loader:
            train_images = train_images.view(-1, 28 * 28)
            optimizer.zero_grad()
            train_outputs = model_1(train_images)
            loss = criterion(train_outputs, train_labels.long())
            loss.backward()
            optimizer.step()
            # Calculating and accumulating loss
            agg_train_loss += loss.item()
            _, predicted = torch.max(train_outputs, dim=1)
            agg_train_correct += (predicted == train_labels).sum()
        train_losses_1.append(agg_train_loss / len(train_loader))
        train_accuracies_1.append(agg_train_correct / len(train_data))
        with torch.no_grad():
            for (test_images, test_labels) in test_loader:
                test_images = test_images.view(-1, 28 * 28)
                test_outputs = model_1(test_images)
                test_predictions = torch.argmax(test_outputs, dim=1)
                loss = criterion(test_outputs, test_labels)
                agg_test_loss += loss.item()
                _, predicted = torch.max(test_outputs, dim=1)
                agg_test_correct += (test_predictions == test_labels).sum()
            test_losses_1.append(agg_test_loss / len(test_loader))
            test_accuracies_1.append(agg_test_correct / len(test_data))

    # Plotting accuracy and loss graphs
    plot_convergence_over_epochs(train_accuracies_1, test_accuracies_1, epochs=EPOCHS, mode='Accuracy', model="1V3")
    # plot_convergence_over_epochs(train_losses_1, test_losses_1, epochs=EPOCHS, mode='CE Loss', model=1)
    # KL
    kl_div_value = kl_divergence_from_nn(model=model_1)
    kl_div_values.append(kl_div_value)
    kl_div_per_param_values.append(kl_div_value / count_parameters(model_1))
    # print(f"Model {1}'s KL Divergence: {kl_div_values[0]}; Per parameter: {kl_div_per_param_values[0]}")
    torch.save(model_1, 'model1V3.pkl')
    return {"model": model_1, "kl_val": kl_div_value, "train_losses": train_losses_1,
            "train_accuracies": train_accuracies_1}
