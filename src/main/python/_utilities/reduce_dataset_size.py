#
#   This a tool for modifying collections of files organized as TF datasets
#
#   Purpose
#       Create smaller samplings of larger sets by skipping REDUCE_FACTOR-1 images at a time.
#

import os
import shutil
from pathlib import Path

# TODO: reduce_factor = Skip all but one out of this many files.
#                       So '10' would means keep 1/10th of the files.
REDUCE_FACTOR = 32
# REDUCE_FACTOR = 128

# TODO: should we erase an existing destination folder?
OVERWRITE = True

# TODO: replace with path to your local path to the source dataset.
#       The folder containing TRAIN + TEST above class folders.

# local_source_path = '~/_Workspace/Datasets/KaggleCartoon/cartoon_classification'
local_source_path = '~/Desktop/Workspace/Datasets/KaggleCartoon/cartoon_classification'
source_path = Path( os.path.expanduser( local_source_path ) )

# TODO: replace with path to your local path to the destination dataset.
# local_destination_path = '~/_Workspace/Datasets/KaggleCartoonReduced/cartoon_classification'
local_destination_path = '~/Desktop/Workspace/Datasets/KaggleCartoonReduced/cartoon_classification'
destination_path = Path( os.path.expanduser( local_destination_path ) )


# perform folder analysis
train_image_count = len(list(Path(source_path/"TRAIN").rglob('*.jpg')))
print("Source Train Image Count=",train_image_count)
print("Estimated Destination Train Image Count=",(int)((train_image_count+REDUCE_FACTOR-1)/REDUCE_FACTOR))

test_image_count = len(list(Path(source_path/"TEST").rglob('*.jpg')))
print("Source Test Image Count=",test_image_count)
print("Estimated Destination Test Image Count=",(int)((test_image_count+REDUCE_FACTOR-1)/REDUCE_FACTOR))

train_class_count = len( os.listdir( source_path / "TRAIN" ) )
test_class_count = len( os.listdir( source_path / "TEST" ) )
print( "\nTrain Class Count=",train_class_count)
print( "Test Class Count=",test_class_count)


#
#   Erase or Create destination folder
#
def rmdir(directory):
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()

if (os.path.exists( destination_path )):
    print("Destination Folder already exists ",destination_path)
    if not OVERWRITE :
        print("Overwrite set to False, exiting process")
        exit(0)
    elif not destination_path.is_dir():
        print("Destination is not a folder, exiting process")
        exit(0)
    else:
        print("Removing destination folder in order to rewrite")
        rmdir( destination_path )

#
# creating folder with TRAIN and TEST
#
os.makedirs( destination_path )

print( "\nDestination Folder Created ",destination_path )

def copy_folder( src_path, dest_path, folder_name ):

    print("Copying ",folder_name," folder")

    index = 0
    from_dir = src_path / folder_name
    to_dir = dest_path / folder_name
    os.mkdir( to_dir )

    # create class folders
    for class_dir in os.listdir( from_dir ):
        os.mkdir( to_dir / class_dir )

    # copy reduced list of files
    for file_path in Path(from_dir).rglob('*.jpg'):
        if (index%REDUCE_FACTOR)==0:
            cname = file_path.parent.name
            fname = file_path.name
            new_file = to_dir / cname /  fname
            shutil.copy2( file_path, new_file )
        index += 1

copy_folder( source_path, destination_path, "TRAIN" )
copy_folder( source_path, destination_path, "TEST" )
print("Copy Complete")

train_image_count = len(list(Path(destination_path/"TRAIN").rglob('*.jpg')))
print("\nActual Destination Train Image Count=",train_image_count)

test_image_count = len(list(Path(destination_path/"TEST").rglob('*.jpg')))
print("\nActual Destination Test Image Count=",test_image_count)
