from torch.utils.data import Dataset
from torch import Generator
from torchvision.transforms.functional import crop
import torch
import rioxarray as rxr
import os
from params import *

class OrthoMosaics(Dataset):
    """
    A dataset object that enables the loading and potentially additional preprocessing of the orthomosaics and the height maps.

    Params:
    --------
    path : str
        The path to the root directory of the FORTRESS dataset. This directory should contain a directory 
        called 'orthomosaics' containing the drone images and a directory called 'nDSM' with the height maps.
    mode : "train" | "validation" | "test"
        Indicates if it is used as a train, test, or validation set. Will use dataset splits according to the 
        fractions defined by HP_TRAIN_FRACTION and HP_TEST_FRACTION defined in `params.py`
    split_seed : int
        random seed to determine which image files to use for a dataset. Should be the same value for 
        train/test/validation sets per model.
    """
    def __init__(self, path: str, mode: str, split_seed: int, crop_dims: tuple | int) -> None:
        super().__init__()
        assert(mode == "train" or mode == "test" or mode == "validation")

        self.path_root = path
        inputs  = [image for image in os.listdir(path + "orthomosaics/") if image not in EXCLUSION_LIST]
        targets = [image for image in os.listdir(path + "nDSM/") if image not in EXCLUSION_LIST]

        # Check for missing files
        self.images = list(set(inputs).union(set(targets)))
        
        self.n_images = len(self.images)

        # Get dataset part according to 'mode'.
        rng = Generator().manual_seed(split_seed)
        img_indices = torch.randperm(self.n_images, generator = rng)

        if(mode == "train"):        self.img_indices = img_indices[0 : int(HP_TRAIN_FRACTION * self.n_images)]
        elif(mode == "test"):       self.img_indices = img_indices[int(HP_TEST_FRACTION * self.n_images) : -1]
        elif(mode == "validation"): self.img_indices = img_indices[int(HP_TRAIN_FRACTION * self.n_images) : int(HP_TEST_FRACTION * self.n_images)]
        else:                       raise ValueError("Mode should be \"train\", \"test\", or \"validation\".")
        

        # Load all images into memory.
        # orthos = torch.zeros((45, 3, 2600, 2600), dtype = torch.float16)
        """ 
        orthos = torch.zeros((self.n_images, 3, IMG_HEIGHT, IMG_WIDTH), dtype = torch.float16)
        ndsms  = torch.zeros((self.n_images, 1, IMG_HEIGHT, IMG_WIDTH), dtype = torch.float16)
        for i in self.img_indices:
            ortho = rxr.open_rasterio(path + "orthomosaics/" + self.images[i]).to_numpy()
            orthos[i,:,:,:] = torch.tensor(ortho / IMG_ORTHO_SCALEFACTOR, dtype = torch.float16)

            ndsm  = rxr.open_rasterio(path + "nDSM/" + self.images[i]).to_numpy()
            ndsms[i,:,:,:]  = torch.tensor(ndsm / IMG_NDSM_SCALEFACTOR, dtype = torch.float16)

        print("Orthos")
        print(orthos.max(), orthos.min())

        print("\n\nndsms")
        print(ndsms.max(), ndsms.min())

        orthos = orthos.to(DEVICE)
        ndsms = ndsms.to(DEVICE)
        print("Doner") 
        """

        # Load both orthomosaic and ndsm images into a single tensor, 
        # where the first 3 channels are the RGB channels of the mosaics, 
        # and the last channel is the ndsm height map.
        # This has as main benefit that data augmentation operations will 
        # have effect on the height map as well, without having to resort to 
        # tricks to apply the same random operation to mosaic and ndsm seperately.
        self.ortho_ndsm = torch.zeros((self.n_images, 4, IMG_HEIGHT, IMG_WIDTH), dtype = torch.float16)
        for i in self.img_indices:
            ortho = rxr.open_rasterio(path + "orthomosaics/" + self.images[i]).to_numpy()
            ndsm  = rxr.open_rasterio(path + "nDSM/" + self.images[i]).to_numpy()

            self.ortho_ndsm[i,0:3,:,:] = torch.tensor(ortho / IMG_ORTHO_SCALEFACTOR, dtype = torch.float16)
            self.ortho_ndsm[i,3,:,:]   = torch.tensor(ndsm  / IMG_NDSM_SCALEFACTOR,  dtype = torch.float16)

        self.ortho_ndsm = self.ortho_ndsm.to(DEVICE)
        print(self.ortho_ndsm.shape)

        """ # Random crop can be implemented in the augmentation layers of the models.
        # Random crop implementation: select a single random crop location within each image.
        # 
        # Random number generator for random crop locations
        self.crop_rng = Generator()
        self.crop_limits = (IMG_HEIGHT - crop_dims[0], IMG_WIDTH - crop_dims[1])
        """
        pass

    def __getitem__(self, index):
        """
        This function should return a single data sample, aka an input image and the target height map.
        """

        """ Load a single mosaic and height map for each index.
        name = self.images[self.img_indices[index]]
        # (3, 2600, 2600)
        ortho = rxr.open_rasterio(self.path_root + "orthomosaics/" + name).to_numpy()
        # normalize images to [0,1]
        ortho = torch.tensor(ortho / 255, dtype = torch.float16)

        # (1, 2600, 2600)
        ndsm  = rxr.open_rasterio(self.path_root + "nDSM/" + name).to_numpy()
        # normalize images to [0,1]
        ndsm  = torch.tensor(ndsm / 255, dtype = torch.float16) \
        """

        # Returns orthomosaic and height map in a single tensor.
        return self.ortho_ndsm[index]
    
    def __len__(self):
        """
        Returns the size of the dataset in number of samples.
        """
        return self.n_images