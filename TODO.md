# Large Goals:

Create cartoon transform using Adventure Time cartoons.
Run 'attention is all you need' locally
Run 'memory enhanced neural networks' locally

Create map generator for turn based puzzle games
Create map solver for turn based puzzle games
Create quality evaluator for turn based puzzle games

## Next Tasks:

cyclegan_runner, update loss to add a value based on the opposite discriminator positively identifying the image
        ( eg.  The horse discriminator thinking that the altered horse image is still a horse )
        ( this might address the issue of white horses not being altered to look like zebras )

methods for loss that let you set the goals

add 'variance' measure, something that checks there is not too much self similarity
    could simply use a zip algorithm ... hmm
--------------------------------------------------------------------------------------
find memory enhanced neural network example from 
https://analyticsindiamag.com/memory-enhanced-neural-networks-might-be-the-next-big-thing/

use one-hot representation and rules to create strategic maps ala original strategic conquest
train to play on such maps ?

use rules when training NN
weight agnostic NN ( evolutionary )

can we reuse the patterns from a conv2d layer in a conv2dtranspose layer ?

highway pattern -> resnet pattern -> unet pattern
the highway pattern in the cyclegan makes the 'new' image look a LOT like to original
is there a way to focus more on the abstracted evaluation ( similar to DallE conversion to text )
I would like for a human figure to look more like a cartoon figure ( three finger hands et al. )

thinking of a kind of criss cross pattern:
    downsample(image), downsample(cartoon), upsample(image), upsample(cartoon)
    where we end up with 4 networks:  d(i)/u(i), d(c)/u(c), d(i)/u(c), d(c)/u(i)

create a StochasticPool2D method which picks a random value from the pool space ( see MaxPool2D ).

create one cartoon classifier example for both ImageDataGenerator and GeneratorBasedBuilder
    determine if ImageDataGenerator is streaming content ( permits for larger datasets )

implement an 'epoch_save' which records epoch information and statistical history
    as well as the model weights.

cyclegan things to try:
    resnet [done/fail]
    dense resnet ( use down/up sample code but do 2 or 3 layer skips )
    dense unet ( same as unet, but more dense than convolution and no skipping )
    terse unet ( same as unet, but use skip=4)
    double unet ( start with terse unet, fold twice )
    unet/resnet criss cross

## Completed Tasks:

example of loading images from video
found dataset with human figures in common settings ( oxford human interaction )

examples of using directly loaded datasets
examples of using ImageDataGenerator for datasets ( does this stream? )
    it seems to let me use larger datasets without memory issues

found cartoon datasets and more at Kaggle

examples of custom datasets
examples of custom loss function

