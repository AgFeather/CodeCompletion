# Code Completion System

A Code completion System with CNN+LSTM

For a given JavaScript source code file with several holes(each hole means several tokens has been taken away),
This system can predict what kind of tokens shold be filled in each hole. 

## Tools
1. python3.6
2. tensorflow1.8.0

## Dataset
A small JavaScript source code dataset.
## Benchmark
1. Create a table M, Count how many times a token occurs following another token. The row token means current token i, column token means following token j. M(i,j) means how many times token j follows token i.
2. For any given test token, we predict the next token of it, we will check the table and return the most fluent token appearing after the test token.

## DNN
Implementation a MLP model to compare the performance.


 
