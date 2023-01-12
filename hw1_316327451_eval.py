import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch
import glob
from hw1_316327451_train import *
from hw1_316327451_train import get_MNIST


# -- utils --
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


# ---Evaluations--

def evaluate_models(model, only_200=False, random_labels=False, only_3_8=False):
    with torch.no_grad():
        test_accuracies, test_losses = [], []
        model.eval()
        agg_test_loss, agg_test_correct = 0, 0
        criterion = nn.CrossEntropyLoss()

        # kl_div_per_param_values = (model["kl_val"] / count_parameters(model["model"]))
        # print(f"Model's KL Divergence: {model['kl_val']}; Per parameter: {kl_div_per_param_values}")

        if only_3_8:
            train_data, _, test_data, test_loader = get_MNIST([3, 8])
        else:
            train_data, _, test_data, test_loader = get_MNIST()


    for (test_images, test_labels) in test_loader:
        test_images = test_images.view(-1, 28 * 28)
        test_outputs = model(test_images)
        test_predictions = torch.argmax(test_outputs, dim=1)
        loss = criterion(test_outputs, test_labels)
        agg_test_loss += loss.item()
        _, predicted = torch.max(test_outputs, dim=1)
        agg_test_correct += (test_predictions == test_labels).sum()
    test_losses.append(agg_test_loss / len(test_loader))
    test_accuracies.append(agg_test_correct / len(test_data))
    print(test_accuracies[-1].item())


######## create training Graphs ########

graph_of_models = {
    # "model1": train_model1(),
    # "model2": train_model2(),
    # "model3": train_model3(),
    # "model4": train_model4(),
    # "model5": train_model5(),  # Random labels
    # "logistic_regression": train_logistic(L2_regularization=False),
    # "logistic_regression_L2_reg": train_logistic(L2_regularization=True),
    "model1_v2": train_model1V2(),
    "model1_v3": train_model1V3()
}
######### Print KL ########
for i, model in enumerate(glob.glob("model?.pkl")):
    loaded_model = torch.load(model)
    kl_div_value = kl_divergence_from_nn(model=loaded_model)
    kl_div_per_param_values = (kl_div_value / count_parameters(loaded_model))
    print(f"Model {i + 1}'s KL Divergence: {kl_div_value}; Per parameter: {kl_div_per_param_values}")

###### accuricies ########


loaded_model = torch.load("model_logistic_nonreg.pkl")
print("model model_logistic_nonreg accuracy")
evaluate_models(loaded_model)
loaded_model = torch.load("modellogistic_l2.pkl")
print("model model logistic with l2 Reg accuracy")
evaluate_models(loaded_model)
loaded_model = torch.load("model1.pkl")
print("model besian V1 accuracy")
evaluate_models(loaded_model)
loaded_model = torch.load("model1v2.pkl")
print("model besian V2 accuracy")
evaluate_models(loaded_model)
loaded_model = torch.load("model1v3.pkl")
print("model besian V3 accuracy")
evaluate_models(loaded_model)
