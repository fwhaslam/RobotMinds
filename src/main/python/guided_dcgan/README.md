What is this?
=============

Watching the GAN example of generating MNIST digits got me to thinking.

The generated images where 'number-like', but were rarely numbers.
I want to try adding digit information to the data, see if that produces 
images that look more like the actual numbers.

Here are the changes:  

01 = same as original, with some cleanup on model + image saving

02 = samples include an embedded representation of the digit value ( eg. 0-9 )

03 = some percent of the samples will have an incorrect and y_pred=false

04 = model decides on a 'claim' for the digit value ( eg. 0-9 )

11 = same as 01, with tighter model using only (2x2) convolutions with extra layers

12 = same as 02, using tighter models
 
13 = same as 03, using tighter models

14 = same as 04, using tighter models

_runAll = runs all scripts three times, and the 'some-false' scripts 3*5 times with rates of ( 10,25,50,76,90 )
        The result is an animation with 150 generations
