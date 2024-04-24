

# import libraries
import rioxarray as rxr
from rasterio.enums import Resampling
from rasterio.warp import reproject
#from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from params import DIR_DATA, EXCLUSION_LIST


from src.setup import make_folders


def preprocess():
    raw_data_dir, processed_data_dir, output_dir = make_folders(Path.home() / Path(DIR_DATA))

    # define paths
    raw_ndsm_files = raw_data_dir / 'nDSM'
    processed_ndsm_files = processed_data_dir / 'nDSM'

    # process all nDSM TIFF files
    for ndsm_file in raw_ndsm_files.glob('*.tif'):
        base_name = ndsm_file.stem.split('_')[1]
        
        if(base_name not in EXCLUSION_LIST): 
            # read each nDSM file
            ndsm = rxr.open_rasterio(ndsm_file, band_as_variable = True)

            # assign O values as NaN to the nDSM files
            ndsm['band_1'] = ndsm['band_1'].where(ndsm['band_1'] != np.float32(-3.4e+38), 0)

            # set the _FillValue attribute
            fill_value_attrs = {'_FillValue': 0}
            ndsm['band_1'].attrs.update(fill_value_attrs)

            # save modified nDSM files to disk
            ndsm.rio.to_raster(processed_ndsm_files / (base_name + ".tif"))
            print(f"{ndsm_file.name} saved to disk at {processed_ndsm_files}")


    # before executing this:
    # exclude CFB044 and CFB089 from orginial orthomosaics and nDSM

    # define paths
    ndsm_path = raw_data_dir / 'nDSM'
    ortho_path = raw_data_dir / 'orthomosaics'
    processed_ortho_files = processed_data_dir / 'orthomosaics'

    # iterate over the orthomosaic files
    for ortho_file in ortho_path.glob('*.tif'):

        base_name = ortho_file.stem.split('_')[0]
        if(base_name not in EXCLUSION_LIST): 
            # determine the corresponding nDSM file
            ndsm_file_name = 'nDSM_' + base_name + '.tif'
            ndsm_file = ndsm_path / ndsm_file_name

            #if not ndsm_file.exists():
                
            #   print(f"Corresponding nDSM file not found for {ortho_file.name}")
            #   continue

            # read the nDSM and orthomosaic files
            ndsm = rxr.open_rasterio(ndsm_file, band_as_variable = True)
            ortho = rxr.open_rasterio(ortho_file, band_as_variable = True)

            # calculate downscale factors
            downscale_factor_x = ndsm.rio.width / ortho.rio.width
            downscale_factor_y = ndsm.rio.height / ortho.rio.height
            
            # calculate new heights and widths
            x_downscaled = round(ortho.rio.width  * downscale_factor_x)
            y_downscaled = round(ortho.rio.height * downscale_factor_y)
            
            # downsample the orthomosaic
            ortho_downsampled = ortho.rio.reproject(
                ortho.rio.crs,
                shape=(int(y_downscaled), int(x_downscaled)),
                resampling=Resampling.bilinear
            )

            # remove the fourth band (no information)
            if 'band_4' in ortho_downsampled:
                
                ortho_downsampled = ortho_downsampled.drop_vars('band_4')

            # write the processed orthomosaic to disk
            ortho_downsampled.rio.to_raster(processed_ortho_files / (base_name + ".tif"))
            print(f"{ortho_file.name} with dimensions ({x_downscaled}, {y_downscaled})")

    # for reading a single file

    # data reading
    #--------------
    # define path to raw nDSM files
    ndsm_path = processed_data_dir / 'nDSM'
    ortho_path = processed_data_dir / 'orthomosaics'

    # read one example file for both nDSM and orthomosaic
    ndsm = rxr.open_rasterio(ndsm_path / 'CFB030.tif', band_as_variable = True)
    ortho = rxr.open_rasterio(ortho_path / 'CFB030.tif', band_as_variable = True)\
    
    print(ndsm)
    print(ortho)

def print_dataset():
    base_dir = Path.home() / Path(DIR_DATA)
    ndsm_path = base_dir / 'processed_data' / 'nDSM'
    ortho_path = base_dir / 'processed_data' / 'orthomosaics'

    # read one example file for both nDSM and orthomosaic
    ndsm = rxr.open_rasterio(ndsm_path / 'CFB030.tif', band_as_variable = True)
    ortho = rxr.open_rasterio(ortho_path / 'CFB030.tif', band_as_variable = True)\
    
    print(ndsm)
    print(ortho)

    print(ndsm.as_numpy().shape)
    print(ortho.as_numpy().shape)

    
