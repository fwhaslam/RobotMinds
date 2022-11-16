What is this?
=============

Watching the GAN example of generating MNIST digits got me to thinking.

The generated images where 'number-like', but were rarely numbers.
I want to try adding digit information to the data, see if that produces 
images that look more like the actual numbers.

Here is the change:  

01 = same as original, with some cleanup on model + image saving

02 = samples include an embedded representation of the digit value ( eg. 0-9 )

03 = some percent of the samples will have an incorrect and y_pred=false

