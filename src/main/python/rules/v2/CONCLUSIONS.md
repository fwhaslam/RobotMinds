2023-11-10
======================

I reduced the problem to two values: ratio and certainty

At first it looked like it was learning fast ( lower 'loss' quickly ).
But it turned out that it was optimizing to improve 'certainty' only.

I tried again by removing certainty, but it did not improve much at 
all after that.  'Ratio' as a global feature does not appear to be 
directly learnable.

I am going to create an intermediate training layer that is simply 
a progression of 100 neurons that are sequentially on or off depending 
on the ratio I am trying to express.   If that is trainable, then I 
will try adding some 'randomizing' logic past that to get something 
more interesting which still has the ratio.
