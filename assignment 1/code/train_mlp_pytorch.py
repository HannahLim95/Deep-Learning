"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    _, predicted = torch.max(predictions, 1)
    _, targeted = torch.max(targets, 1)
    accuracy = (predicted == targeted).sum().item() / len(predicted)

    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def train():
    """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    # Get negative slope parameter for LeakyReLU
    neg_slope = FLAGS.neg_slope

    ########################
    # PUT YOUR CODE HERE  #
    #######################


    #default settings
    # nr_iterations = FLAGS.max_steps + 1
    # learning_rate = LEARNING_RATE_DEFAULT  # best learing-rate
    # hidden_layers = dnn_hidden_units
    # neg_slope = FLAGS.neg_slope
    # batch_size = FLAGS.batch_size

    # optimal settings
    nr_iterations = 6000 + 1
    # learning_rate = 2e-3 # optimal learning rate for SGD
    learning_rate = 1e-4 # optimal learning rate for Adam
    # weight_decay = 1e-4 # optimal weight decay for SGD
    weight_decay = 1e-5  # optimal weight decay for Adam
    hidden_layers = [1000, 500, 200, 100, 100]
    neg_slope = 0.002
    batch_size = FLAGS.batch_size

    cifar10 = cifar10_utils.get_cifar10()
    n_classes = 10
    n_inputs = 32 * 32 * 3

    mlp = MLP(n_inputs, hidden_layers, n_classes, neg_slope)
    loss = nn.CrossEntropyLoss()

    # default optimizer
    # optimizer = optim.SGD(mlp.parameters(), lr=learning_rate)

    # optimal optimizer settings
    # # optimizer = optim.SGD(mlp.parameters(), lr=learning_rate, weight_decay=1e-4)
    optimizer = optim.Adam(mlp.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)

    accuracies_test = []
    accuracies_train = []
    losses = []
    for i in range(nr_iterations):
        x, y = cifar10['train'].next_batch(batch_size)
        x = torch.from_numpy(x.reshape(-1, n_inputs))
        y = torch.from_numpy(y).type(torch.LongTensor)
        optimizer.zero_grad()
        prediction = mlp(x)
        _, y_target = y.max(1)
        cross_entropy_loss = loss(prediction, y_target)
        losses.append(cross_entropy_loss.item())
        cross_entropy_loss.backward()
        optimizer.step()
        if i % FLAGS.eval_freq == 0:
            x_test, y_test = cifar10['test'].images, cifar10['test'].labels
            x_test = torch.from_numpy(x_test.reshape(-1, n_inputs))
            y_test = torch.from_numpy(y_test).type(torch.LongTensor)
            pred_test = mlp(x_test)
            acc_test = accuracy(pred_test, y_test)
            acc_train = accuracy(prediction, y)
            print('accuracy at step', i, ': ', acc_test)
            accuracies_test.append(acc_test * 100)
            accuracies_train.append((acc_train*100))

    # print(accuracies_test)
    # print(accuracies_train)
    # print(losses)

    # plotting the accuracies for the train and test set
    # plt.subplot(2, 1, 1)
    # plt.plot(np.arange(len(accuracies_test) * FLAGS.eval_freq, step=FLAGS.eval_freq), accuracies_test, label ='test data')
    # plt.plot(np.arange(len(accuracies_train) * FLAGS.eval_freq, step=FLAGS.eval_freq), accuracies_train, label = 'train_data')
    # plt.axhline(y=52, color='green', linestyle='dashed', label='target line')
    # plt.ylabel('Accuracy (%)')
    # plt.legend()
    # plotting the losses from the train set
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(np.arange(len(losses)), losses)
    # plt.xlabel('Steps')
    # plt.ylabel('Loss')
    #
    # plt.savefig('pytorch_MLP')
    # plt.show()

    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
  Prints all entries in FLAGS variable.
  """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
  Main function
  """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                        help='Negative slope parameter for LeakyReLU')
    FLAGS, unparsed = parser.parse_known_args()

    main()
