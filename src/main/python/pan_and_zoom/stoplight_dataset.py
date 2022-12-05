#
#   Base Dataset is Lisa Traffic Lights
#
#   Examples are street images that sometimes contain traffic lights.
#   The sampling will flip, randomized screen regions, randomized zoom, and randomized brightness
#
#

import sys
sys.path.append('..')

# common
import os as os
from pathlib import Path

import tensorflow as tf
import numpy as np

import csv
import random
import weakref
import math

from log_tools import *
from collections import OrderedDict
import matplotlib.pyplot as plt

# TODO: modify path to match local user-space
ROOT_PATH = Path( os.path.expanduser( 'D:/Datasets/LisaTrafficLight/data' ) )

VIEW_WIDE = 32
VIEW_TALL = 32
CHANNELS = 3
IMAGE_SHAPE = ( VIEW_WIDE, VIEW_TALL, CHANNELS )

OPTIONS_COUNT = 31
OPTIONS_SHAPE = ( OPTIONS_COUNT, )

lastFigure = None

########################################################################################################################
# box management

class box_rec:

    def __init__(self, filepath, tag, ulcx, ulcy, lrcx, lrcy ):

        self.filekey = filepath.split("/")[-1]      # source filename
        self.tag = tag                              # light state

        self.ulcx = ulcx                            # light box
        self.ulcy = ulcy
        self.lrcx = lrcx
        self.lrcy = lrcy

    def key(self):
        return self.filekey


class box_parser:

    def __init__(self, header ):
        self.filenameIx = header.index("Filename")
        self.tagIx = header.index("Annotation tag")
        self.ulcxIx = header.index("Upper left corner X")
        self.ulcyIx = header.index("Upper left corner Y")
        self.lrcxIx = header.index("Lower right corner X")
        self.lrcyIx = header.index("Lower right corner X")

    def parse( self, row ):

        return box_rec( row[self.filenameIx],
                        row[self.tagIx],
                        row[self.ulcxIx],
                        row[self.ulcyIx],
                        row[self.lrcxIx],
                        row[self.lrcyIx] )


class fbox:
    r"""Box/Rectangle with floating bounds."""

    def __init__( self, ulcx, ulcy, lrcx, lrcy ):
        self.ulcx = float(ulcx)
        self.ulcy = float(ulcy)
        self.lrcx = float(lrcx)
        self.lrcy = float(lrcy)
        return

    def __str__(self):
        return "fbox( {}, {}, {}, {} )".format( self.ulcx, self.ulcy, self.lrcx, self.lrcy )

    def overlap(self,vs):
        r"""Find the overlap box.  None if there is none."""
        # print('self=',self,'  vs=',vs)
        # print('self.type=',type(self.ulcx))
        # print('vs.type=',type(vs.ulcx))
        new_ulcx = max( self.ulcx, vs.ulcx )
        new_ulcy = max( self.ulcy, vs.ulcy )
        new_lrcx = min( self.lrcx, vs.lrcx )
        new_lrcy = min( self.lrcy, vs.lrcy )

        if (new_ulcx>new_lrcx or new_ulcy>new_lrcy): return None

        return fbox( new_ulcx, new_ulcy, new_lrcx, new_lrcy )

    def pixels(self):
        r"""How many pixels does this box occupy?"""
        return (self.lrcx-self.ulcx) * (self.lrcy - self.ulcy)

    def horz(self,shift):
        dist = (self.lrcx-self.ulcx) * shift
        return fbox( self.ulcx+dist, self.ulcy, self.lrcx+dist, self.lrcy )

    def vert(self,shift):
        dist = (self.lrcy-self.ulcy) * shift
        return fbox( self.ulcx, self.ulcy+dist, self.lrcx, self.lrcy+dist )

    def zoom(self,shift):
        r"""Positive zoom makes the box smaller.  Negative zoom makes it larger."""
        dist2 = (self.lrcx-self.ulcx) * shift / 2.
        return fbox( self.ulcx-dist2, self.ulcy+dist2, self.lrcx-dist2, self.lrcy-dist2 )

    def as_tensor(self):
        r"""Tensor 'box', with y before x for use with tf.crop_and_resize()"""
        return tf.constant( [self.ulcy,self.ulcx,self.lrcy,self.lrcx])


def box_to_fbox( box:box_rec ):
    return  fbox( box.ulcx, box.ulcy, box.lrcx, box.lrcy )

########################################################################################################################
#   Model Actions ( eg. Options )

def sv_func( horz:float, vert:float, zoom:float ):
    def shift_view( view:fbox ):
        work = view.horz(horz)
        work = work.vert(vert)
        return work.zoom(zoom)
    return shift_view

OPTION_FUNCTIONS = [
    sv_func(0.05,0.,0.), sv_func(0.1,0.,0.), sv_func(0.2,0.,0.), sv_func(0.4,0.,0.), sv_func(0.8,0.,0.),
    sv_func(-0.05,0.,0.), sv_func(-0.1,0.,0.), sv_func(-0.2,0.,0.), sv_func(-0.4,0.,0.), sv_func(-0.8,0.,0.),

    sv_func(0.,0.05,0.), sv_func(0.,0.1,0.), sv_func(0.,0.2,0.), sv_func(0.,0.4,0.), sv_func(0.,0.8,0.),
    sv_func(0.,-0.05,0.), sv_func(0.,-0.1,0.), sv_func(0.,-0.2,0.), sv_func(0.,-0.4,0.), sv_func(0.,-0.8,0.),

    sv_func(0.,0.,0.05), sv_func(0.,0.,0.1), sv_func(0.,0.,0.2), sv_func(0.,0.,0.4), sv_func(0.,0.,0.8),
    sv_func(0.,0.,-0.05), sv_func(0.,0.,-0.1), sv_func(0.,0.,-0.2), sv_func(0.,0.,-0.4), sv_func(0.,0.,-0.8),

    sv_func( 0.,0.,0. )
]

# These options are what the model will select, they are named and described here.
# Note that we cannot emulate changing focus with this dataset, just zoom.
# 5p = 5 percent
# 31 options total
OPTION_NAMES = [

    "pan_horz_left_5p","pan_horz_left_10p","pan_horz_left_20p","pan_horz_left_40p","pan_horz_left_80p",
    "pan_horz_right_5p","pan_horz_right_10p","pan_horz_right_20p","pan_horz_right_40p","pan_horz_right_80p",

    "pan_vert_up_5p","pan_vert_up_10p","pan_vert_up_20p","pan_vert_up_40p","pan_vert_up_80p",
    "pan_vert_dn_5p","pan_vert_dn_10p","pan_vert_dn_20p","pan_vert_dn_40p","pan_vert_dn_80p",

    "zoom_in_5p","zoom_in_10p","zoom_in_20p","zoom_in_40p","zoom_in_80p",
    "zoom_out_5p","zoom_out_10p","zoom_out_20p","zoom_out_40p","zoom_out_80p",

    "stop"
]

########################################################################################################################
#   Weak References to large images

class stoplight_dataset:
    r"""This is not a Tensorflow Dataset.
    This is just a class that returns Image sets with annotations"""

    rnd = random.Random()

    def __init__(self,*args,**kwargs):

        self._args = args
        self.file_map = self.load_file_map()
        self.box_map = self.load_box_map()
        self.key_list = list( self.file_map.keys() )
        self.weak_image_map = {}        # images are stored using weak references
        return

    def load_image(self,file):
        r"""Load image from hard drive"""
        return tf.keras.utils.load_img( file )

    def save_image(self, key, image ):
        self.weak_image_map[key] = weakref.ref( image )
        return

    def find_image( self, key ):
        r"""Image may exist locally.  If not then load from hard drive."""
        if key in self.weak_image_map:
            image = self.weak_image_map[key]()
            if not image is None:
                return image

        file = self.file_map[key]
        image = self.load_image( file )
        # print("IMAGE TYPE=",type(image))
        self.save_image( key, image )

        return image

########################################################################################################################

    def load_file_map(self):

        log.debug("loading file map" )
        dict = {}
        for file in ROOT_PATH.rglob('*.jpg'):
            if "sample" in str(file):    # skip the sample folders
                continue
            key = file.name
            if key in dict:     # duplicates are a problem
                raise Exception( 'Duplicate file key [{}] from path[{}] vs path[{}]'.format(key,file,dict[key]) )
            dict[ key ] = file
        log.debug("loading file map complete" )

        return dict

    def load_box_map(self):

        log.debug("loading box map" )
        dict = OrderedDict()
        for file in ROOT_PATH.rglob('*BOX.csv'):

            if "sample" in str(file):    # skip the sample folders
                continue

            # reset column index
            with open( file, newline='\n' ) as csvfile:

                reader = csv.reader( csvfile, delimiter=';' )
                parser = None

                for row in reader:
                    if parser is None:
                        parser = box_parser( row )
                        continue

                    rec = parser.parse( row )
                    key = rec.key()

                    if not key in dict: dict[key] = []
                    dict[ rec.filekey ].append( rec )

        log.debug("loading box map complete" )
        return dict

########################################################################################################################

    def build( self, samples=10 ):
        r"""Create a number of samples as follows:
        Samples are a rectangular selection from an image, reduced to 32x32.
        The ground truth ( y_true ), is an array of options with a value assigned.
        The ground truth is compared to a one-hot array which selects one option.
        The loss is based on the value of the ground truth which matches the selection."""

        images = np.empty( (samples,)+IMAGE_SHAPE )
        notes = np.empty( (samples,)+OPTIONS_SHAPE )

        for index in range(0,samples):
            image,note = self.make_sample()
            images[index] = image
            notes[index] = note

        return images, notes

    def make_sample(self):
        r"""Return an image View and a 'y_true' that has one value for each Option."""

        key = self.rnd.choice( self.key_list )
        # print('key=',key)
        image = self.find_image( key )

        box_list = None
        if key in self.box_map:
            box_list = self.box_map[ key ]
        # print('box_list=',box_list)


        shape = tf.cast( tf.shape( image ), tf.float32 )
        # print('shape=',shape)
        # shape[0] = vert, shape[1] = horz
        inner = fbox( 0., 0., shape[1], shape[0] )
        # print('inner=',inner)

        view = self.pick_view( inner )
        # print('view=',view)

        option_values = self.calc_option_values( view, box_list )
        view_image = self.calc_view_image( image, view )

        return view_image, option_values

    def pick_view( self, inner ):

        rmin = 8
        rmax = int( min( inner.lrcx, inner.lrcy ) ) - 1
        rsize = self.rnd.randint( rmin, rmax )

        offx = self.rnd.randint( 0, int( inner.lrcx - rmax) )
        offy = self.rnd.randint( 0, int( inner.lrcy - rmax) )
        return fbox ( offx, offy, offx+rsize, offy+rsize )


    def calc_option_values(self, view, box_list):

        start_rating = self.overlap_percent( view, box_list )

        values = []
        for option_fnc in OPTION_FUNCTIONS:
            option_view = option_fnc( view )
            option_rating = self.overlap_percent( option_view, box_list )
            values.append( option_rating - start_rating )

        # slight extra weight to 'stop'
        values[-1] = values[-1] + 0.00001

        return values


    def overlap_percent(self, view, box_list ):

        pixels = 0.

        if not box_list is None:
            for box in box_list:
                newbox = box_to_fbox(box)
                # print("newbox=",newbox)
                # print('newbox.type=',type(newbox.ulcx))
                lap = newbox.overlap( view )
                if not lap is None:
                    pixels = lap.pixels()

        # percent of view pixels in view covered by stoplight box
        pixels_in_view = pixels / view.pixels()

        #TODO: percent of box
        return pixels_in_view

    def show_image(self, title, image ):
        r"""For testing purposes."""
        global lastFigure
        plt.ion()
        plt.close( lastFigure )
        lastFigure = plt.figure( title )
        plt.imshow( image )
        plt.pause(30)
        return

    def calc_view_image( self, source, view ):
        # self.show_image( 'original', source )
        # tf.print('source.shape=',tf.shape(source))
        # tf.print('view=',str(view))
        image = tf.image.crop_to_bounding_box(
            source,
            int(view.ulcy), int(view.ulcx),
            int(view.lrcy-view.ulcy), int(view.lrcx-view.ulcx) )
        # tf.print('cropped=',image)
        # self.show_image( 'cropped', image )
        image = tf.cast( image, tf.float32 ) / 255.
        # tf.print('scaled=',image)
        # self.show_image( 'scaled', image )
        image = tf.image.resize( image, (VIEW_WIDE,VIEW_TALL) )
        # tf.print('resized=',image)
        # self.show_image( 'resized', image )
        return image

# work = pnz_base_dataset()
#
# # print('filemap len=', len( work.file_map ))
# # print('boxmap len=', len( work.box_map ))
# #
# #
# # size = len([f for f in work.file_map])
# # print("base image count=",size)
# #
# # size = sum( (f in work.box_map) for f in work.file_map )
# # print("images with lights=",size)
# #
# # size = sum( (not f in work.box_map) for f in work.file_map )
# # print("images with ZERO lights=",size)
# #
# # size = sum( ((f in work.box_map) and len(work.box_map[f])==1 ) for f in work.file_map )
# # print("images with ONE lights=",size)
# #
# # size = sum( ((f in work.box_map) and len(work.box_map[f])>1 ) for f in work.file_map )
# # print("images with MANY lights=",size)
#
# images,notes = work.build( 1 )
