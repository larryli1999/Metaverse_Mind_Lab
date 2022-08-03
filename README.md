# Metaverse_Mind_Lab

This is the take home assignment for Metavese Mind Lab. The dataset .csv files are not included in the repo since those exceed the maximum size limit.

## Scripts expalined:

```train.py```: pre-process the training data, convert text data to TF-IDF features and train the model. 

```pre_process.py```: store all the pre-process functions need for the text

```predict.py```: load the trained model, pre-process the test data and labels and run prediction/ evaluation on those test data

## Result:
- Applied 70/30 data split for the training data to ensure that the model does not overfit.
- Implement logistic regression as the classification model on the generated TF-IDF features
- Achieve 95% average F1-score for both class on dev set
- Achieve 63% average F1-score for both class on test set

## Future improvements:
- Experiment with different model architectures to see which improves the test performance
- Apply different word embeddings such as word2vec or glove
- Experiment techqiues such as cross-validation to see if those solve the problem
- Explore details about the test vs. training dataset
