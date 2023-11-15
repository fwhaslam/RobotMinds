
import os
from pathlib import Path

# for var in ('HOME', 'USERPROFILE', 'HOMEPATH', 'HOMEDRIVE'):
#     print( var +'=' + str( os.environ.get(var) ) )

ROOT_FOLDER = os.path.expanduser( '~/_Workspace/Datasets/KaggleCartoon/cartoon_classification' )

def _generate_examples(folder_path ):
    """Generator of examples for each split."""
    for img_path in folder_path.glob('*.jpg'):
        # Yields (key, example)
        yield img_path.name, {
            'image': img_path,
            'label': img_path.parent.name,
        }


data_path = Path( ROOT_FOLDER ) / 'TEST'

print("PATH=",data_path)

for f in os.listdir( data_path ):
    print("FILE=",f)


generator = _generate_examples( data_path )

for i in generator:
    print(i)