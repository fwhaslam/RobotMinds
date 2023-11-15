#
#   Cartoon Dataset from Kaggle.
#       from: https://www.kaggle.com/datasets/volkandl/cartoon-classification
#
import os.path
import tensorflow_datasets as tfds
from pathlib import Path
import tensorflow as tf
import glob
import tensorflow.keras.utils as tfku


class KaggleCartoonDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for my_dataset dataset."""

    VERSION = tfds.core.Version('0.0.1')
    RELEASE_NOTES = {
        '0.0.1': 'Initial implementation.',
    }

    # TODO: you need to download the dataset, add it to your local folder structure, then changes this value
    ROOT_FOLDER = '~/_Workspace/Datasets/KaggleCartoon/cartoon_classification'

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(180, 320, 3)),
                # 'image': tfds.features.Image(shape=(360, 640, 3)),
                'label': tfds.features.ClassLabel(
                    names=['adventure_time', 'catdog', 'Familyguy', 'Gumball', 'pokemon',
                           'smurfs', 'southpark', 'spongebob', 'tom_and_jerry', 'Tsubasa']
                )
            }),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Download the data and define splits."""
        #extracted_path = dl_manager.download_and_extract('http://data.org/data.zip')

        extracted_path = Path( os.path.expanduser( self.ROOT_FOLDER ) )
        # dl_manager returns pathlib-like objects with `path.read_text()`,
        # `path.iterdir()`,...
        return {
            'train': self._generate_examples( folder_path = extracted_path / 'TRAIN' ),
            'test': self._generate_examples( folder_path = extracted_path / 'TEST' ),
        }

    def _generate_examples(self, folder_path ):
        count = -1
        """Generator of examples for each split."""
        for img_path in folder_path.rglob('*.jpg'):
            count += 1
            if ( count % 16 ) != 0:
                continue
            # Yields (key, example)
            yield ( img_path.parent.name +'/' + img_path.name ), {
                'image': img_path,
                # 'image': tf.image.resize( tfku.load_img( img_path ), ( 360, 640 ) ),
                'label': img_path.parent.name,
            }