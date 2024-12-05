# Exercise 9

This is the ninth exercise for the course [Deep Learning](https://lehre.idh.uni-koeln.de/lehrveranstaltungen/wintersemester-2024-2025/deep-learning/).

Clone this repository to your local setup. Create a branch with the name of your GitHub account and switch to it.

In the repository, you find a CSV file with first names originating from different nationalities.

Train a RNN and a BiRNN with Keras to predict the origin of a name. Train on 80% of the dataset and test on the remaining 20%.
Use and implement several of the training strategies presented in the [lecture](https://lehre.idh.uni-koeln.de/site/assets/files/5372/presentation08.pdf).
You will need to adapt the Keras Tokenizer to tokenize characters instead of words (https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer)

Let the model predict on the test set and compute Accuracy, Precision, Recall and F1-score.
Try to get an F1 score as high as possible.

Also compare your results with using a FFNN instead of a RNN.

Commit your changes and push them to the branch with the name of your GitHub account.

Deadline for this exercise is December 12, 2024, 08:00:00 CET.
