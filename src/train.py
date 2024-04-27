from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot, softmax
from torch.nn import Module
import torch.optim as optim

from datetime import datetime
import json
import numpy as np
import os

from params import *
from src.utils import print_train_progress, print_validation_progress, print_final
from src.models import ARCHITECTURES, Augmentation, Cropper
from src.dataloader import OrthoMosaics


class EarlyStopping():
    def __init__(self, min_delta: float = 0.5, limit = 3) -> None:
        self.min_delta = min_delta
        self.limit = limit
        self.cnt = 0
    
    def stop(self, val_loss, train_loss):
        if((val_loss - train_loss) > self.min_delta):
            self.cnt +=1
            if(self.cnt >= self.limit):
                return True
        else:
            self.cnt = 0
            return False


# Train model for a single epoch
def train(
    loader: DataLoader, 
    model: Module,
    preprocessing: Module,
    optimizer,
    epoch,
    epochs,
):
    losses = []
    
    model.train()
    n_batches = len(loader)
    for k, img in enumerate(loader):
        # img = img.to(DEVICE).float()
        optimizer.zero_grad()
        
        img = preprocessing(img)
            # img = aug[:, 0:3, :, :]
            # target = aug[:, 3:4, :, :]
            
        
        # Forward
        target, pred = model(img)
        
        loss = model.loss_func(pred, target)
        losses.append(loss.item())

        # Backward
        loss.backward()
        model.float()
        optimizer.step()
        model.half()

        print_train_progress(epoch, epochs, k, n_batches, loss.item())
    return losses

def validate(
    loader: DataLoader, 
    model: Module,
    preprocessing: Module,
    epoch,
    epochs
):
    losses = []
    
    model.eval()

    n_batches = len(loader)
    for k, img in enumerate(loader):
        # img = img.to(DEVICE).float()
        
        img = preprocessing(img)
        # Forward
        target, pred = model(img)
        
        loss = model.loss_func(pred, target)
        losses.append(loss.item())
        
        print_validation_progress(epoch, epochs, k, n_batches, loss.item())
    return losses

def test(
    loader: DataLoader, 
    model: Module,
    preprocessing: Module
):
    losses = []
    
    model.eval()

    for k, img in enumerate(loader):
        # img = img.to(DEVICE).float()
        
        img = preprocessing(img)
        # Forward
        target, pred = model(img)
        
        loss = model.loss_func(pred, target)

        losses.append(loss.item())
        
    return losses

def train_model(
    name: str,
    arch_name: str,
    epochs: int,
    image_crop,
    batch_size: int,
    lr,
    split_seed = 42,
    lr_schedule = False,
    early_stopping = False,
    description: str = None
):
    """
    Train a model.
    
    Params:
    ----------
        name : str
            Name of this model. A new model will be created if there is no metric and model file present in their respective directories.
        arch_name : str
            Name of the architecture defined in models.py. Only used when generating a new model.
        epochs : int
            Number of epochs to train.
        lr : Any
            learning rate
        batch_size : int
            Number of training samples in a batch.
        description : str
            (Optional) A string with a short description about the model.
    """

    print("Training model \"{}\" based on the \"{}\" architecture".format(name, arch_name))

    model_exists = (name + ".pt") in os.listdir(DIR_MODELS) and (name + ".json") in os.listdir(DIR_METRICS)
    
    if(model_exists):
        # load model
        model = torch.load(DIR_MODELS + name + ".pt").to(DEVICE)

        # Load metric file
        with open(FILENAME.format(DIR = DIR_METRICS, NAME = name, EXT = ".json"), 'r') as f:
            report = json.load(f)
        
        augmentation = Augmentation(report["input shape"]).to(DEVICE)
        cropper      = Cropper(report["input shape"]).to(DEVICE)

        # Resume training
        current_epochs = report["params"]["epochs"]
        if(current_epochs >= epochs): 
            print("No more epochs to train.\nExiting.")
            return
    
    else:
        # Generate a new model from arch_name
        architecture = ARCHITECTURES[arch_name]
        model = architecture(image_crop).to(DEVICE)
        augmentation = Augmentation(image_crop).to(DEVICE)
        cropper      = Cropper(image_crop).to(DEVICE)

        # Generate metric file
        report = {
            "params":{
                "architecture": arch_name,
                "description": model.name if description == None else description,
                "input shape": image_crop,
                "date": datetime.now().strftime('%Y-%m-%d_%H-%M'),
                "epochs": 0,
                "learning rate": lr,
                "n_params": np.sum(p.numel() for p in model.parameters() if p.requires_grad),
                "split seed": split_seed
            },
            "model structure": model.__str__(),
            "train":[],
            "evaluation":[]
        }
        current_epochs = 0
    
    # Load data
    train_data   = OrthoMosaics(DIR_DATA + "processed_data/", mode = "train", split_seed = split_seed)
    val_data     = OrthoMosaics(DIR_DATA + "processed_data/", mode = "validation", split_seed = split_seed)
    test_data    = OrthoMosaics(DIR_DATA + "processed_data/", mode = "test", split_seed = split_seed)

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    val_loader   = DataLoader(val_data,   batch_size = batch_size, shuffle = True)
    test_loader  = DataLoader(test_data,  batch_size = batch_size, shuffle = True)
    
    # Optimizer
    if(lr_schedule):
        optimizer = optim.Adam(model.parameters(), lr, weight_decay=HP_WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.93, current_epochs - 1)
    else:
        optimizer = optim.Adam(model.parameters(), lr, weight_decay=HP_WEIGHT_DECAY)

    if(early_stopping):
        stopper = EarlyStopping()
    
    try:
        # Repeat training for epochs.
        for i in range(current_epochs, epochs):
            # Train
            train_losses = train(train_loader, model, augmentation, optimizer, i, epochs)

            if(lr_schedule):
                scheduler.step()
            
            val_losses = validate(val_loader, model, cropper, i, epochs)

            report_epoch = {
                "epoch": i,
                "train_loss": train_losses,
                "val_loss": val_losses
            }
            report["train"].append(report_epoch)
            report["params"]["epochs"] = i


            mean_train_loss = np.mean(train_losses)
            mean_val_loss = np.mean(val_losses)
            print_final(i, epochs, mean_train_loss, mean_val_loss)
            print()

            # Save model and metrics
            with open(FILENAME.format(DIR = DIR_METRICS, NAME = name, EXT = ".json"), 'w') as f:
                json.dump(report, f, indent = 4)
            
            torch.save(model, FILENAME.format(DIR = DIR_MODELS, NAME = name, EXT = ".pt"))
            
            if(early_stopping):
                mean_train_loss = np.mean(train_losses)
                if(stopper.stop(mean_val_loss, mean_train_loss)):
                    print("Stopping training")
                    break

        # Test
        losses = test(test_loader, model, cropper)

        report["test"] = {
            "test_loss": losses
        }

        # Save metrics
        with open(FILENAME.format(DIR = DIR_METRICS, NAME = name, EXT = ".json"), 'w') as f:
            json.dump(report, f, indent = 4)
            
    except KeyboardInterrupt:
        pass

def test_model(name: str):
    model_exists = (name + ".pt") in os.listdir(DIR_MODELS) and (name + ".json") in os.listdir(DIR_METRICS)
    if(not model_exists):
        print("Model does not exist")
    
    model = torch.load(DIR_MODELS + name + ".pt").to(DEVICE)
    with open(FILENAME.format(DIR = DIR_METRICS, NAME = name, EXT = ".json"), 'r') as f:
        report = json.load(f)
    
    # Data loader
    data = OrthoMosaics(DIR_DATA, "test", report["params"]["split seed"])
    test_loader  = DataLoader(data, batch_size=report["params"]["batch_size"])

    losses = test(test_loader, model)

    report["test"] = {
        "test_loss": losses
    }
    with open(FILENAME.format(DIR = DIR_METRICS, NAME = name, EXT = ".json"), 'w') as f:
        json.dump(report, f, indent = 4)

