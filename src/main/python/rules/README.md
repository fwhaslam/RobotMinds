What is this?
=============

The concept is simple.  I want to train the NN using rules instead of examples.

What I have in mind is to tell it about maps for a game.  Instead of giving examples, 
I will apply rules that judge the quality of the output.  Will this be an effective 
technique?  I have no idea.  So far so good.

Step One / v1: land and sea = generate a map similar to old school strategy maps, 
    with just two types of terrain.  The rules specify the ratios I want to see,
    and the amount of 'surface' (eg. coastline )

Step Two / v2: land, sea, deep, peaks = four types of terrain
Terrain has a ratio of volumes  The surface goals are defined for terrain pairs.  
eg.  sea to peak should be zero, land to deep should be zero, etc.
Dropping some of the less successful models.
Templates will be constructed based on color similarity between image and ONE_HOT_COLORS

v3: side step, simplifying to a single 'land ratio' value, 
    and a random grid for input.  The goal is to make sure 
    that the ratio of produced land vs sea matches the ratio.

v4: use feature 'ratio', and training set of sequential neurons 
    activated by ratio. ( eg.  ratio 0.5 = first 50 active, second 50 inactive )
    Try first with on/off activation.  Try again with softmax.
    See if either is trainable, and/or form affects function.

attention / embedding ?
NOTE: attention can be used to reorder things, 
    which in two dimensions means transform an image ...?

https://medium.com/geekculture/no-data-no-problem-3561a08f35c5

try these things:
1) more linear activation:  relu -> LeakyReLu [Small]
1) more normalized activation:  relu -> selu [Major]
1) merge all lower layers together at each step [None]
1) feature range = -0.5 to 0.5 [None] ( weird, still failing to 'fill' towards end of range ...)

Something strange here, feature is BOTH [0,1] and [1,0] paired.
So we would expect both ends of the result to have equal results,
BUT one end is consistently not changing.
    Committing so i can try on another system.

1) feature range an exponential ranged value
1) integral feature with embedding for representation
1) rgb output instead of one-hot ( treated like embedding )
1) generated examples
1) stable diffusion
1) Attention layer reorders tokens, can it be extended to 2 dimensions ?
