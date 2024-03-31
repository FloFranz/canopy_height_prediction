import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, top_k_accuracy_score
import numpy as np
import json
from params import *



    
def plot_loss_evaluation(name):
    with open(DIR_METRICS + name + ".json", 'r') as f:
        report = json.load(f)

    if("model structure" in report):
        print(report["model structure"])
    print(report["params"])
    
    train_loss = [np.mean(epoch["train_loss"]) for epoch in report["train"]]
    val_loss = [np.mean(epoch["val_loss"]) for epoch in report["train"]]
    
    if("test" in report):
        print("Test loss: ", np.mean(report["test"]["test_loss"]))
    
    fig, ax1 = plt.subplots(figsize=(6.5, 5))
    fig.suptitle(name)
    l1 = ax1.plot(range(1, len(train_loss) + 1), train_loss, 'r', label = 'train loss')
    l2 = ax1.plot(range(1, len(val_loss)+ 1), val_loss, 'g', label = 'validation loss')
    ax1.set_ylim(0, 2)
    # plt.plot(val_accuracy, 'b', label = 'validation accuracy')
    # for x in epoch_indices: plt.axvline(x, color = 'r')
    # plt.vlines(epoch_indices, 0.0, 1.0, color = 'r')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel("Loss")
    ax1.set_xticks(range(1, len(train_loss) + 1))
    ax1.legend()

    # fig.set_size_inches(7, 4)
    fig.savefig(DIR_METRICS + name + "_eval_loss.pdf")

def print_loss_report(filename):
    with open(DIR_METRICS + filename + ".json", 'r') as f:
        report = json.load(f)
    print("Epoch:\tTrain loss:\tValidation loss:")
    for epoch in report["train"]:
        e = epoch["epoch"]
        tl = np.mean(epoch["train_loss"])
        vl = np.mean(epoch["val_loss"])
        print(f"{e:2d}\t{tl:.4f}\t\t{vl:.4f}")
    

if __name__ == "__main__":
    # get_weirdness()
    name = "test_model"

    

    plot_loss_evaluation(name, report)