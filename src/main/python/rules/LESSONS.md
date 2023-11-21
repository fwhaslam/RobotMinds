Important Lessons Learned:

1) A variable feature is learned better than a fixed feature.  
    I found that a sliding 'ratio' feature as an input was learned better  
    than a fixed value which was imposed by the rules.

1) Screwing up the 'shuffle' can lead to lopside datasets which 
   prevent learning.

1) Selu gives better results than most other activation functions, 
    probably because it removes the need for normalization.

1) a test assertion which ensures tensor shape was EXTREMELY useful.

1) ~~features with value range [0-1], are learned better with a
   second inverted input ( so [0-1], and [1-0] )~~  
   Tested with 4 terrain types, with and without inverted values,  
   At 50 epochs there was a slight improvement in loss, BUT  
   at 250 epochs the difference was marginal.
