# What Happened?


### Process

This is all eyeball mark one.  There are no metrics for measuring GAN performance.
    However, the 'some-false' variations could measure 'accuracy' for the discriminator,
    and conversely the 'accuracy' for generated samples.


### Evaluations

01/original = produces images that can be called 'number-like'.  Some are recognizable as
    numbers ( digits ), others are just abstract objects that look sorta like a number.
    Digits are distinct by epoch 50, and only drift a little between then and 150.

11/tighter = tighter layers means more, smaller layers.  Original is using 5x5 convolutions,
    tighter is mostly using 2x2 convolutions.  
    Performs about as well as 'original'.  Digits are distinct by epoch 50, 
    drift a little up to 150.  At epoch 35 the original is fuzzy, while 'tighter' is grainy.

02/identity = every sample presented to discriminator has a label ( digit ).  
    The generator is given a digit that it needs to produce.
    At epoch 50 digits are mostly distinct, some are still abstract.  The digits
    are approaching what the label says.  At 100 more digits are distinct.
    At 150 all but one of the 16 samples is distinctly representing the correct digit.

12/tighter+identity = Every sample has a label, and using tighter layers.
    At epoch 50, the results are worse than 02/identity.  At 100 all but the 'threes' 
    look like their label.  At 150 everything looks like their digit, although 
    one two and one three look soemwhat damaged.

03/some-false = the label stays the same, but some percentage of the images are modified
    so they should not be a match for the label.
    At 10%, seems to be about the same as 'original'
    At 50%, performs about as well as 'original'.  Perhaps a slight increase in drift for
    last 100 epochs.  The number-like shapes might be slightly better.
    At 90%, number-like objects still form, similar to 'original'.  The epochs after 50 
    tend to drift more. The shape is less stable than for original.
  
13/tighter+some-false = Some samples have false images, the model uses tight layers.
    This seems to perform about the same as 03/some-false, except that there is 
    slightly more graininess and less fuzziness.

04/claim = The generator creates its own label, eg. a 'claim' about what it represents.
    Note that the model is designed so that the claim comes as part of the first layer, 
    and is then fed through the model. The claimed number will change over the course of epochs.
    Epoch 50 is about as good as 'original', except that a little over half the numbers 
    resemble their claims.  There is significant drift over then next 100 epochs.
    Epoch 150 is not much bettern than epoch 50.  Perhaps half the numbers look like their
    claims, and another quarter resemble their claims.

14/tighter+claim = Generates claims, uses tighter model layers.
    Epoch 50 looks to be worse than 04/claim at epoch 50.
    There is significant drift over the next 100 epochs.
    Epoch 100 is about as good as the 04/claim epoch 50.
    Epoch 150 everything looks approximately like its claim.


## Lessons and Thoughts

For this task, the broader convolution layers in the discriminator look to perform better 
as part of an Adversarial Network. I don't know if this is because it is better at discriminating
or because it is worse.  I can just conclude that fuzzier discriminators works better for this GAN.

I briefly tried using the tighter generator and fuzzier discriminator.  Performance more closely 
resembled the original ( eg. the same or better ).

Passing in an 'identity' label to the generator produced more distinctive images.
The results all resembled the digit from the label.  There were much fewer 'number-like' images.

'Claims' did not work as well as 'Identity'.  Perhaps if the claim was generated from the output 
instead of the input then it would work better.  It may be that the unchanging value of identity 
was a much better anchor than the flexible claim value.

I was pleased with using 'selu' instead of LeakyReLU+Normalizataion.

