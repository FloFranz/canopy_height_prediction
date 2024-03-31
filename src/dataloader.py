from torch.utils.data import Dataset

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
    def __init__(self, path, mode, split_seed) -> None:
        super().__init__()
        pass

    def __getitem__(self, index):
        """
        This function should return a single data sample, aka an input image and the target height map.
        """
        return None, None
    
    def __len__(self):
        """
        Returns the size of the dataset in number of samples.
        """
        return 0