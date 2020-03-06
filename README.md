# egg_chick_intervals
Contains program code for logistic regression on number of eggs and chicks as a function of predictors, when these numbers can be intervals. The inference tool is Bayesian, using MCMC. Random variables and fixed factorialvariables are allows. These should start with "rfactor_" and "ffactor_" respectively. The program is called with a csv file and the set of predictors in that file denominated with + or minus the column number, plus for egg variables and minus for chick variables. The file should contain four columns labeled "egg.start", "egg.end", "kids.start" and "kids.end" to specify the start and end of the egg/chick intervals. (If start and end is the same, it means the number is fixed). It is assumed that a maximum of eggs is 7, but this should be easy to change in the code. A prior file can be given, containing the predictor number, range and the upper limit of the 95% credibility interval for the multiplication factor that the explanation variable can affect the odds. 

A help text is shown when calling the program without input arguments. This also shows the available options.

The program uses the hydrasub library, found at http://folk.uio.no/hydrasub. This library will later be moved to Github.
