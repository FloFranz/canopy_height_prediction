#-------------------------------------------------------------
# Name:         setup.py
# Description:  script sets up a working environment,
#               defines file paths for data import and output.
#-------------------------------------------------------------

from pathlib import Path

def make_folders(base_dir = Path.cwd().parent):
    
    # 01 - setup working environment
    # --------------------------------

    # create directory called 'data' with subdirectories
    # 'raw_data', 'processed_data', and 'metadata'
    # base_dir = Path.cwd().parent
    # base_dir = Path(root_dir)

    data_directories = [
        base_dir / 'data' / 'raw_data' / 'nDSM',
        base_dir / 'data' / 'raw_data' / 'orthomosaics',
        base_dir / 'data' / 'processed_data' / 'nDSM',
        base_dir / 'data' / 'processed_data' / 'orthomosaics',
        base_dir / 'data' / 'metadata'
    ]

    for directory in data_directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f'created directory {directory.relative_to(base_dir)}')

    # create other necessary directories
    other_directories = [
        base_dir / 'output'
    ]

    for directory in other_directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f'created directory {directory.relative_to(base_dir)}')

    # 02 - file path definitions
    # ---------------------------

    # define raw data directory
    raw_data_dir = base_dir / 'data' / 'raw_data'

    # define processed data directory
    processed_data_dir = base_dir / 'data' / 'processed_data'

    # define output directory
    output_dir = base_dir / 'output'

    return raw_data_dir, processed_data_dir, output_dir
