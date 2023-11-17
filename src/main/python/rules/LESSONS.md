Important Lessons Learned:

1) A variable feature is learned better than a fixed feature.  
I found that a sliding 'ratio' feature as an input was learned better  
than a fixed value which was imposed by the rules.

1) Screwing up the 'shuffle' can lead to lopside datasets which  
prevent learning.

1) Selu gives better results than most other activation functions,  
probably because it removes the need for normalization.

