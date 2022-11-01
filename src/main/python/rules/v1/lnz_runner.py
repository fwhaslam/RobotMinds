#
#   Larger Goal is to create a map suitable for use in Strategic Conquest with
#   six terrains:  sea, land, rough, neutral city, white city, black city
#
#
#   The goal of this design is to produce one-hot map of land + sea
#   goodness is measure via analysis of continuity and surface area ( fractal edge )
#
#   The input data is going to be a mix of terrain maps, classical art, and random noise
#
#   see:
#       https://www.kaggle.com/datasets/tpapp157/terrainimagepairs
#       https://www.kaggle.com/datasets/tpapp157/earth-terrain-height-and-segmentation-map-images
#       https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time
#

import sys
sys.path.append('../..')

# common
import os as os
import numpy as np
import random as rnd
from pathlib import Path
# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# tensorflow
import tensorflow as tf
from tensorflow import image, keras

from land_and_sea_functions import *
from _utilities.tf_tensor_tools import *
from cycle_gan.tf_layer_tools import *

# tf.compat.v1.enable_eager_execution()

lastFigure = None
# BATCH_SIZE = 1
# EPOCHS = 5

IMAGE_CHANNELS = 3
WIDE = TALL = 32
INPUT_SHAPE = (WIDE, TALL, IMAGE_CHANNELS)
MAP_SHAPE =  (WIDE, TALL, TERRAIN_TYPE_COUNT)
IMAGE_RESIZE = list(INPUT_SHAPE[:2])     #  [wide,tall]


class lnz_runner:

    def __init__(self,*args,**kwargs):

        self._args = args

        self.flavor = args[0]
        self.model:tf.keras.Model = args[1]
        self.loss_function = args[2]
        self.optimizer = args[3]

        self.BATCH_SIZE = 1
        if 'batch_size' in kwargs: self.BATCH_SIZE = kwargs['batch_size']

        self.EPOCHS = 5
        if 'epochs' in kwargs: self.EPOCHS = kwargs['epochs']

        plt.ion()


########################################################################################################################
#   Display Stuff


    def describe_work( self, work ):
        r"""Produce string that evaluates the image"""

        # print('work=',work)
        # if True: return "hiya\nboo"

        # print('type_loss=', terrain_type_loss( work ) )
        # print('certainty_loss=', terrain_certainty_loss( work ) )
        # print('surface_loss=', terrain_surface_loss( work ) )

        # transform into final map
        work = tensor_to_value( tf.argmax( tf.round( work ), axis=-1 ) )
        # print("final work=",work)

        # count terrain types
        ttype = [0.] * TERRAIN_TYPE_COUNT
        for i in range(0, WIDE):
            for j in range(0, TALL):
                index = work[i][j]
                ttype[ index ] = ttype[ index ] + 1

        max = WIDE * TALL
        # print('\nTerrain Counts=',ttype,
        #       'goal=',tensor_to_value(TERRAIN_TYPE_GOAL),
        #       'ratio=', tensor_to_value( tf.constant(ttype) / max ) )
        type_ratio = tensor_to_value( tf.constant(ttype) / max )

        # count transitions
        wide1 = WIDE-1
        tall1 = TALL-1
        ttrans = [0.] * TERRAIN_TYPE_COUNT
        for i in range(0, WIDE):
            for j in range(0, TALL):
                index = work[i][j]

                count = 0
                if index != work[(i+wide1)%WIDE][j]: count = 1+count
                if index != work[i][(j+tall1)%TALL]: count = 1+count
                if index != work[(i+1)%WIDE][j]: count = 1+count
                if index != work[i][(j+1)%TALL]: count = 1+count

                ttrans[ index ] = ttrans[ index ] + count

        max = 2 * WIDE * TALL
        # print('Terrain Transitions=',ttrans,
        #       'goal=',tensor_to_value( TERRAIN_SURFACE_GOAL ),
        #       'ratio=', tensor_to_value( tf.constant(ttrans) / max ) )
        surface_ratio = tensor_to_value( tf.constant(ttrans) / max )

        # print()
        return  "Type"+str(type_ratio)+"\nSurface"+str(surface_ratio)


    def to_map_image( self, image_values, one_hot_color ):

        onehot = image_values
        # tf.print('work/squeeze=',tf.shape(work))

        onehot = tf.round( onehot )
        # tf.print('work/round=',work)
        onehot = tf.cast(onehot, tf.int32)
        # tf.print('work/cast=',work)

        return tf.tensordot( onehot, one_hot_color, 1 )


    def draw_image_and_values( self, values, display ):
        r"""Work is an array of images"""

        # # values = tf.squeeze(work)
        # values = work
        #
        # # two colors
        # display = self.to_map_image( values, TERRAIN_ONE_HOT_COLOR )

        global lastFigure
        plt.close( lastFigure )
        title = self.flavor+'@'+self.model.name+ \
                ' t'+str(tensor_to_value(TERRAIN_TYPE_GOAL))+ \
                ' s'+str(tensor_to_value(TERRAIN_SURFACE_GOAL))
        lastFigure = plt.figure( title, figsize=(9, 10) )

        plt.rcParams.update({
            'font.size': 8,
            'text.color': 'black'})

        for index in range(9):
            sub = plt.subplot( 3, 3, 1+index )
            sub.title.set_text( self.describe_work( values[index] ) )
            plt.axis('off')
            plt.imshow( display[index] )

        # tf.print("values=",values.shape)
        # if (WIDE<=32):
        #     for (j,i,t),value in np.ndenumerate(values):
        #         # tf.print('shape=',value.shape)
        #         if (t==0):
        #             label = int( 10 * value )
        #             plt.text(i,j,"{}".format(label),ha='center',va='center')

        plt.show()
        plt.pause( 500 )
        return

    def draw_some_images( self, work, title ):
        r"""Work is an array of images"""

        global lastFigure
        plt.close( lastFigure )
        title = title+' for '+self.flavor+'@'+self.model.name+ \
                ' t'+str(tensor_to_value(TERRAIN_TYPE_GOAL))+ \
                ' s'+str(tensor_to_value(TERRAIN_SURFACE_GOAL))
        lastFigure = plt.figure( title, figsize=(9, 10) )

        plt.rcParams.update({
            'font.size': 8,
            'text.color': 'black'})

        for index in range(9):
            sub = plt.subplot( 3, 3, 1+index )
            # sub.title.set_text( self.describe_work( values[index] ) )
            sub.title.set_text( f'index({index})' )
            plt.axis('off')
            plt.imshow( work[index] )

        # tf.print("values=",values.shape)
        # if (WIDE<=32):
        #     for (j,i,t),value in np.ndenumerate(values):
        #         # tf.print('shape=',value.shape)
        #         if (t==0):
        #             label = int( 10 * value )
        #             plt.text(i,j,"{}".format(label),ha='center',va='center')

        plt.show()
        plt.pause( 500 )
        return


    def display_results( self, test_images, test_image_set ):

        # print('DR test_images.shape=',test_images)
        # print('DR test_image_set.shape=',test_image_set)
        # if True: return

        # if the set has two different shapes, then only use the first part
        # print("test_image_set TYPE = ",type(test_image_set))
        if isinstance( test_image_set, list ):
        # if test_image_set[0].shape != test_image_set[1].shape:
            test_image_set = test_image_set[0]


        # pick and display one solution
        images = test_images[0:9]
        templates = test_image_set[0:9]

        result = self.model( images )

        # if the set has two different shapes, then only use the first part
        # print("result TYPE = ",type(result))
        if isinstance( result, list ):
        # if result[0].shape != result[1].shape:
            result = result[0]

        # print('result=',result)
        # if True: sys.exit()

        # double output case means we need to drop the second result
        # [work,pic] = result
        # work = result
        if isinstance(result, list):
            work_one_hot = result[0]
        else:
            work_one_hot = result

        # if len(result)>1: work = result[0]
        # tf.print('work.shape=',work.shape)
        # tf.print('pic.shape=',pic.shape)
        # if True: exit(0)

        self.draw_some_images( images, 'originals' )

        template_as_images = self.to_map_image( templates, TERRAIN_ONE_HOT_COLOR )
        self.draw_some_images( template_as_images, 'templates' )

        display = self.to_map_image( work_one_hot, TERRAIN_ONE_HOT_COLOR )
        self.draw_image_and_values( work_one_hot, display )

########################################################################################################################

    def run(self,
            train_images,train_image_set,
            test_images,test_image_set,
            ckpt_folder):

        # loss_function = terrain_loss
        # optimizer = tf.keras.optimizers.Adam(0.001)
        #
        # train_image_set = train_images
        # test_image_set = test_images
        #
        # checkpoint_folder = "landnsea_ckpt"
        # model = create_model_v1(INPUT_SHAPE)

        self.model.summary()

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=['accuracy'] )

########################################################################################################################
# load / train model

        # tflt.try_load_weights( CHECKPOINT_FILE, model )

        # print( "TRAIN SHAPE=",tf.shape(train_images) )
        # print( "Variables.Module=",tf.Module.trainable_variables )
        # print( "Variables.model=",model.trainable_variables )

        self.model.fit(
            x=train_images, y=train_image_set,
            epochs=self.EPOCHS,
            validation_data=(test_images,test_image_set),
            batch_size=self.BATCH_SIZE )

        result = self.model.evaluate( x=test_images, y=test_image_set,  verbose=2 )

        tf.print('result=',result)
        # print('\ntest_loss:', test_loss)
        # print('\ntest_acc:', test_acc)

        ckpt_file = ckpt_folder + '/ckpt'
        # model.save_weights( ckpt_file )

        # if not sys.exists(ckpt_folder): os.makedirs(ckpt_folder)
        # if not exists(filename): open(filename, 'a').close()

        self.display_results( test_images, test_image_set )

