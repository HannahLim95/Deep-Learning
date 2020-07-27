"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch.optim as optim
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  cifar10 = cifar10_utils.get_cifar10()
  n_channels = 3
  n_classes = 10
  nr_iterations = FLAGS.max_steps+1
  convnet = ConvNet(n_channels, n_classes).to(device)

  optimizer = optim.Adam(convnet.parameters(), lr=FLAGS.learning_rate)
  loss = nn.CrossEntropyLoss()

  accuracies_test = []
  accuracies_train = []
  losses = []

  for i in range(nr_iterations):
    x, y = cifar10['train'].next_batch(FLAGS.batch_size)  # (batch_size, 3, 32, 32) (batch_size, 10)
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).type(torch.LongTensor).to(device)
    _, y_target = y.max(1)

    optimizer.zero_grad()
    prediction = convnet(x)

    cross_entropy_loss = loss(prediction, y_target)
    losses.append(cross_entropy_loss.item())
    cross_entropy_loss.backward()
    optimizer.step()

    del x, y_target

    if i % FLAGS.eval_freq == 0:
      x_test, y_test = cifar10['test'].images, cifar10['test'].labels
      x_test = torch.from_numpy(x_test).to(device)
      y_test = torch.from_numpy(y_test).type(torch.LongTensor).to(device)
      pred_test = convnet(x_test)
      acc_test = accuracy(pred_test, y_test)
      acc_train = accuracy(prediction, y)
      accuracies_test.append(acc_test )
      accuracies_train.append(acc_train)
      print('accuracy at step', i, ': ', acc_test)
      del x_test, y_test, pred_test, prediction

  # print(accuracies_train)
  # print(accuracies_test)
  # print(losses)

  #plotting the accuracies for the train and test set
  # plt.subplot(2, 1, 1)
  # plt.plot(np.arange(len(accuracies_test) * FLAGS.eval_freq, step=FLAGS.eval_freq), accuracies_test, label='test data')
  # plt.plot(np.arange(len(accuracies_train) * FLAGS.eval_freq, step=FLAGS.eval_freq), accuracies_train,
  #          label='train_data')
  # plt.axhline(y=75, color='green', linestyle='dashed', label='target line')
  # plt.ylabel('Accuracy (%)')
  # plt.legend()

  # plotting the losses from the train set
  # plt.subplot(2, 1, 2)
  # plt.plot(np.arange(len(losses)), losses)
  # plt.xlabel('Steps')
  # plt.ylabel('Loss')
  #
  # plt.savefig('convnet_MLP')
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
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
