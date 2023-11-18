2023/11/17
==========

Since the algorithms were not learning even the simple feature of 'land/sea' ratio,
I decided to simplify to just one feature ( the ratio ) and test data that is 
a straight line of cells filled from zero to one hundred based on the ratio.
So a ratio of 25% would fill the first 25 cells.

Once I got it working, I could see that it was not learning the ratio above 80%.
Upon investigation I discovered that I was using the 'shuffle' function wrong,
so the last 20% of my examples were never sampled for training.

This version I learned to:
1) fix shuffling
2) ensure my shapes ( added support_for_testing.object_has_shape() assertion )
3) figured out that a simple network learns this fine
4) found out that 'selu' was definitely a superios activation function
5) features with value range [0-1], are learned better with a second inverted input ( so [0-1], and [1-0] )
