import numpy as np
from numpy import genfromtxt

GROUP : 25
STUDENT_NAME = ['AKASH KADIRI','AISHWARYA SURESHRAO WAGH','ARANJAY AVINASH REWANWAR']
STUDENT_ID = ['20925765','20909044','20927312']

# defined sigmoid function
def sigmoid_func(x):
    return 1 / (1 + np.exp(-x))
# definded the derivative of sigmoid function
def sigmoid_drv(x):
    return sigmoid_func(x) * (1 - sigmoid_func(x))
# defined softmax fucntion
def softmax_func(x):
    val = np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
    return val

# feed forward function
def fwd(inp_data, wt_hidlyr, bias_hidlyr,wt_outlyr,bias_outlyr):
    net_hidden = np.dot(inp_data, wt_hidlyr) + bias_hidlyr
    act_hidden = sigmoid_func(net_hidden)
    net_output = np.dot(act_hidden, wt_outlyr) + bias_outlyr
    act_output = softmax_func(net_output)
    return act_output, act_hidden, net_hidden

# Convert predicted data into one hot encoding
def one_hot_enc(x):
	for i in range(0,len(x)):
		x[i,x[i,:].argmax()]=1
	out = (x == 1).astype(float)
	return out

def test_mlp(data_file):
	# Load the test set
	# START
	Z_test = genfromtxt(data_file, delimiter=',')
    # END

	# Load your network
	# START
	weight_hidden = np.load('./Updated Weights/weight_hidden.npy')
	bias_hidden = np.load('./Updated Weights/bias_hidden.npy')
	weight_output = np.load('./Updated Weights/weight_output.npy')
	bias_output = np.load('./Updated Weights/bias_output.npy')
	# END

	# y_pred = ...
	y_pred = fwd(Z_test, weight_hidden, bias_hidden, weight_output, bias_output)
	# Predict test set - one-hot encoded
	y_pred = one_hot_enc(y_pred)
	# return y_pred
	return y_pred

'''
How we will test your code:

from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 

y_pred = test_mlp('./test_data.csv')

test_labels = ...

test_accuracy = accuracy(test_labels, y_pred)*100
'''