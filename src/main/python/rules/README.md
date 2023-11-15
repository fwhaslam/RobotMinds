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