import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, top_k_accuracy_score
import numpy as np
import json
import os
from torch import Generator
import rioxarray as rxr
from src.models import Cropper
from torchvision.transforms.v2 import RandomCrop, CenterCrop

from params import *


def plot_prediction(name):
    """
    Load a random image from the dataset and plot the 
    true height map and the model prediction.

    Params:
    --------
    name : str
        Name of the model to be evaluated.
    """
    model_exists = (name + ".pt") in os.listdir(DIR_MODELS) and (name + ".json") in os.listdir(DIR_METRICS)
    
    if(model_exists):
        # load model
        model = torch.load(DIR_MODELS + name + ".pt").to("cpu")
        model.eval()
        # Load metric file
        with open(FILENAME.format(DIR = DIR_METRICS, NAME = name, EXT = ".json"), 'r') as f:
            report = json.load(f)
    else:
        print("Name does not point to an existing metric AND model file.")
        return
    
    # Random area cropper.
    input_cropper = RandomCrop(report["params"]["input shape"])
    
    # Cropper to match input image size to output of model.
    size_matcher =  CenterCrop(model.target_dim)

    # Load a random image
    inputs  = [image for image in os.listdir(DIR_DATA + "processed_data/orthomosaics/") if image not in EXCLUSION_LIST]
    targets = [image for image in os.listdir(DIR_DATA + "processed_data/nDSM/") if image not in EXCLUSION_LIST]
    images = list(set(inputs).union(set(targets)))

    rng = Generator()
    img_index = torch.randint(0, len(images), (1,)).item()
    
    ortho_ndsm = torch.zeros((1, 4, IMG_HEIGHT, IMG_WIDTH))
    ortho = rxr.open_rasterio(DIR_DATA + "processed_data/orthomosaics/" + images[img_index]).to_numpy()
    ndsm  = rxr.open_rasterio(DIR_DATA + "processed_data/nDSM/" + images[img_index]).to_numpy()
    
    ortho_ndsm[0,0:3,:,:] = torch.tensor(ortho / IMG_ORTHO_SCALEFACTOR)
    ortho_ndsm[0,3,:,:]   = torch.tensor(ndsm   / IMG_NDSM_SCALEFACTOR) 
    
    # make prediction
    ortho_ndsm = input_cropper(ortho_ndsm)
    target, pred = model(ortho_ndsm)

    # Move color channels to first dimension
    ortho  = size_matcher(ortho_ndsm[0,0:3,:,:])
    ortho  = np.moveaxis(ortho.numpy(), 0, 2)
    pred   = np.moveaxis(pred[0].detach().numpy(), 0, 2) * IMG_NDSM_SCALEFACTOR
    target = np.moveaxis(target[0].detach().numpy(), 0, 2) * IMG_NDSM_SCALEFACTOR

    # (pred).detach().numpy()
    # Plotting
    fig, (ax_ortho, ax_pred, ax_ndsm) = plt.subplots(ncols = 3)

    ax_ortho.imshow(ortho.astype(np.float32))
    ax_ortho.set_title("Orthomosaic")

    ax_pred.imshow(pred.astype(np.float32))
    ax_pred.set_title("Prediction")

    ax_ndsm.imshow(target.astype(np.float32))
    ax_ndsm.set_title("Height")

    fig.set_size_inches(14, 8)
    fig.savefig(FILENAME.format(DIR = DIR_METRICS, NAME = name, EXT = ".pdf"))
    plt.show()





    

    
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
    ax1.set_ylim(0, 0.1)
    # plt.plot(val_accuracy, 'b', label = 'validation accuracy')
    # for x in epoch_indices: plt.axvline(x, color = 'r')
    # plt.vlines(epoch_indices, 0.0, 1.0, color = 'r')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel("Loss")
    # ax1.set_xticks(range(1, len(train_loss) + 1))
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

    

    plot_loss_evaluation(name)