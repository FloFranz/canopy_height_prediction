import sys
from params import *


def print_train_progress(epoch, epochs, batch, batches, loss):
    t1 = int(float(batch / batches) * (PROGRESS_BAR_TRAIN))
    t2 = PROGRESS_BAR_TRAIN - t1 - 1
    t_bar =  t1 * "=" + ">" + " " * t2
    v_bar = PROGRESS_BAR_VALIDATE * " "
    text = "\33[2K\r{:3d}/{}: |{}|{}| - Loss: {:.4f}    ".format(epoch, epochs, t_bar, v_bar, loss)
    sys.stdout.write(text)
    sys.stdout.flush()

def print_validation_progress(epoch, epochs, batch, batches, loss):
    v1 = int(float(batch / batches) * (PROGRESS_BAR_VALIDATE))
    v2 = PROGRESS_BAR_VALIDATE - v1 - 1
    v_bar =  v1 * "=" + ">" + " " * v2
    t_bar = PROGRESS_BAR_TRAIN * "="
    text = "\33[2K\r{:3d}/{}: |{}|{}| - Loss: {:.4f}   ".format(epoch, epochs, t_bar, v_bar, loss)
    sys.stdout.write(text)
    sys.stdout.flush()

def print_final(epoch, epochs, mean_train_loss, mean_val_loss):
    text = "\33[2K\r{:3d}/{}: |{}|{}| - Loss: {:.4f} - {:.4f}    ".format(epoch, epochs, PROGRESS_BAR_TRAIN * "=", PROGRESS_BAR_VALIDATE * "=", mean_train_loss, mean_val_loss)
    sys.stdout.write(text)
    sys.stdout.flush()