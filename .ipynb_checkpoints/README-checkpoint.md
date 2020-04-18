# Jigsaw Multilingual Toxic Comment Classification

## Domain Background
The Conversation AI team, a research initiative founded by Jigsaw and Google, builds technology to protect voices in conversation.
A main area of focus is machine learning models that can identify toxicity in online conversations, where toxicity is defined as anything
rude, disrespectful or otherwise likely to make someone leave a discussion. If these toxic contributions can be identified, we could
have a safer, more collaborative internet. I chose this project because of my interest in Natural Language Processing.

Jigsaw's API, Perspective, serves toxicity models and others in a growing set of languages. Over the past year, the field has seen
impressive multilingual capabilities from the latest model innovations, including few- and zero-shot learning.

## Problem Statement
The challenge is to build multilingual models with English-only training data. The training data will be the English data while the test
data will be Wikipedia talk page comments in several different languages. The goal is to predict the probability that a comment is toxic.
A toxic comment would receive a 1.0. A benign, non-toxic comment would receive a 0.0.

## Datasets and Inputs
The Files
•	jigsaw-toxic-comment-train.csv - data from the first competition. The dataset is made up of English comments from Wikipedia’s talk page edits.
•	jigsaw-unintended-bias-train.csv - data from the second competition. This is an expanded version of the Civil Comments dataset with
  range of additional labels.
•	sample_submission.csv - a sample submission file in the correct format
•	test.csv - comments from Wikipedia talk pages in different non-English languages.
•	validation.csv - comments from Wikipedia talk pages in different non-English languages.
•	jigsaw-toxic-comment-train-processed-seqlen128.csv - training data preprocessed for BERT
•	jigsaw-unintended-bias-train-processed-seqlen128.csv - training data preprocessed for BERT
•	validation-processed-seqlen128.csv - validation data preprocessed for BERT
•	test-processed-seqlen128.csv - test data preprocessed for BERT


## Columns
•	id - identifier within each file.
•	comment_text - the text of the comment to be classified.
•	lang - the language of the comment.
•	toxic - whether the comment is classified as toxic. (Does not exist in test.csv.)

The link to the competition page can be found here 


## Solution Statement
I will write a helper function to preprocess the raw data. The preprocessed data will be converted to bag of words and then to
word-vector representation. Different algorithms will be tested on the data which will include Neural Networks and XGBOOST. The goal
will be to make the model loss reduce to the minimal while keeping an eye on the evaluation metrics. 

## Benchmark Model
I will keep tuning the model parameters and considering several other alternative approaches while keeping an eye on moving top on the
competition leaderboard score.

## Evaluation Metrics
The evaluation metrics is on area under the ROC curve between the predicted probability and the observed target.

## Project Design
The first step will be to preprocess the data and make it ready for training. Then I will develop a model based on based performing
research papers on similar work. I will train the model on Amazon Sagemaker. I will keep refining the model and fine tuning the parameters
until the day the competition closes. Finally, the model will be deployed and hosted for my friends to play around with.

Tools and Libraries to be used: Python3, Jupyter Notebook, pandas, pytorch, seaborn, matplotlib, Keras. Other libraries will be added if
deemed necessary.

## References

[1]	Wikipedia https://en.wikipedia.org/wiki/Receiver_operating_characteristic
[2]	Kaggle jigsaw comment-classification. https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/
[3]	Udacity Proposal Review Rubrics

https://review.udacity.com/#!/rubrics/410/view
