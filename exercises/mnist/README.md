# MNIST Classifier

Classifies digits sketched by the user, with a model trained with the MNIST dataset.

![screenshot](capture.gif)

When an image is written by the user, the software first preprocesses it similarly to how the MNIST data were preprocessed (converts it to white on black, shrinks it to 20px-20px, centers its center-of-mass on a black 28px-28px image), then obtains a prediction from the model.

The model consists of five individually-trained MLPs, each trained on the MNIST dataset for 10 epochs with minibatch size 200. Each MLP consists of an input layer of 784 units (one for each pixel), a hidden layer of 784 units (relu activation), and a 10-unit output layer (softmax activation).
