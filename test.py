from src.dataloader import OrthoMosaics
from src.preprocessing import preprocess, print_dataset
from params import DIR_DATA

ortho = OrthoMosaics(DIR_DATA + "processed_data/", "train", 42, (240,240))
# preprocess()
# print_dataset()
