# Code Completion

A Code completion System with deep learning

## Tools
1. python3.6
2. tensorflow1.8.0

## Dataset
A small dataset with 1000 JavaScript source code. 800 for training, 200 for testing.

## Benchmark
1. Create a table M, Count how many times a token occurs following another token. The row token means current token i, column token means following token j. M(i,j) means how many times token j follows token i.
2. For any given test token, we predict the next token of it, we will check the table and return the most fluent token appearing after the test token.

## DNN


 
