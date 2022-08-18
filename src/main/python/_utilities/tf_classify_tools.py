#
#   Common Tools to simplify some tasks in tensorflow and python
#

import tensorflow.python.framework.errors_impl as errimp
import builtins
import os
from pprint import pprint

# try to delete local files
def try_delete( file_path ):
    if os.path.exists( file_path) :
        os.remove (file_path )
    else:
        print("Cannot find file to remove ["+file_path+"]" )

# try to load weights for a model, restore default on failure
def try_load_weights( ckpt_file, model ) :
    temp = model.get_weights()
    if os.path.exists(ckpt_file+'.index') :
        try:
            model.load_weights( ckpt_file )
        except ( builtins.ValueError, errimp.NotFoundError ) as e1:
            # print('Error Is '+ str(e1.__class__) )
            # print('Error Is '+ str(e1.__class__.__module__) )
            # print('Error Is '+ str(e1.__class__.__module__.__module__) )
            model.set_weights(temp)
            print("Old model does not match new model, not loading weights")
            # removing default checkpoint files
            try_delete( "checkpoint" )
            try_delete( ckpt_file + ".index" )
            try_delete( ckpt_file + ".data-00000-of-00001" )
        except Exception as e2:
            # print('Error Is '+ str(e2.__class__) )
            # print('Error Is '+ str(e2.__class__.__module__) )
            print('Failed to load weights: '+ str(e2) )
