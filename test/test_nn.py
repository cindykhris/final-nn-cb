# TODO: import dependencies and write unit tests below

from nn.nn import NeuralNetwork
from nn.preprocess import sample_seqs, one_hot_encode_seqs
from nn.io import read_text_file, read_fasta_file

# import sklearn split function
from sklearn.model_selection import train_test_split
import numpy as np
import pytest

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

##############################################################################################################
##############################################################################################################
########################################     Loading      ####################################################
##############################################################################################################
##############################################################################################################

digits = load_digits()  # Load the digits dataset
X, y = digits.data, digits.target  # X is the data, y is the target

X = X/16  # Normalize the data

# Split the data into training and testing sets (use 20% of the data for testing) 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Creatе an instance of the NeuralNetwork class with 64x16x64 autoencoder architecture with defined hyperparameters
nn_arch = [{"input_dim": 64, "output_dim": 16, "activation": "relu"},
           {"input_dim": 16, "output_dim": 64, "activation": "sigmoid"}]

nn = NeuralNetwork(nn_arch, lr=0.001, seed=42, batch_size=3, epochs=100,
                   loss_function='mse')

# Train the autoencoder
train_loss, val_loss = nn.fit(X_train, X_train, X_val, X_val)

##############################################################################################################
##############################################################################################################
########################################     Testing      ####################################################
##############################################################################################################
##############################################################################################################


def test_single_forward():
    # write a unit test to test the _single_forward method in the NeuralNetwork class
    # this method takes in a W_curr: ArrayLike, b_curr: ArrayLike, A_prev: ArrayLike, and activation: str
    # and returns the output of the forward pass for a single layer

    W_curr = nn._param_dict['W1']
    b_curr = nn._param_dict['b1']
    A_prev = X_train[0]

    A_curr, Z_curr = nn._single_forward(W_curr, b_curr, A_prev, 'relu')

    assert A_curr.shape == (16,16)

def test_forward():

    A_curr, Z_curr = nn.forward(X_train[0])
    
    assert A_curr.shape == (16,64)

test_forward()

def test_single_backprop():


    W_curr = nn._param_dict['W1']
    b_curr = nn._param_dict['b1']
    A_prev = X_train[0]

    

    A_curr, Z_curr = nn._single_forward(W_curr, b_curr, A_prev, 'relu')
    A_prev = A_curr


    dZ_curr = nn._relu_backprop(A_curr, Z_curr)

    dW_curr, db_curr, dA_prev = nn._single_backprop(W_curr, b_curr, Z_curr, A_prev, A_curr, 'relu')
    

    assert dW_curr.shape == (64,16)
    assert db_curr.shape == (16,16)
    assert dA_prev.shape == (16,1)

test_single_backprop()

def test_predict():
    
        y_pred = nn.predict(X_train)
        
    
        assert y_pred.shape == (1437,64)

test_predict()

def test_binary_cross_error():

    y_pred = nn.predict(X_train)
    y_true = X_train

    loss = nn._binary_cross_error(y_pred, y_true)

    assert loss.shape == ()
   

test_binary_cross_error()

def test_binary_cross_error_backprop():

    y_pred = nn.predict(X_train)
    y_true = X_train


    dA = nn._binary_cross_error_backprop(y_pred.T, y_true)
    

    assert dA.shape == (64,1437)

test_binary_cross_error_backprop()

def test_mean_squared_error():
    
        y_pred = nn.predict(X_train)
        y_true = X_train

    
        loss = nn._mean_squared_error(y_pred, y_true)
    
        assert loss == 7.276754186796848
        

test_mean_squared_error()

def test_mean_squared_error_backprop():

    y_pred = nn.predict(X_train)
    y_true = X_train


    dA = nn._mean_squared_error_backprop(y_pred.T, y_true)
    

    assert dA.shape == (64,1437)

test_mean_squared_error_backprop()


def test_sample_seqs():
    from nn.preprocess import sample_seqs

    pos_seqs = read_text_file("data/test_pos_seqs.txt")
    neg_seqs = read_fasta_file('data/test_neg_seqs.fa')

    assert len(pos_seqs) == 15
    assert neg_seqs[0][0] == 'C'

    # L:oading the data test_seqs.txt
    file = "/Users/cindybarrios/Desktop/BMI203_algorithms/final-nn-cb/data/test_data.txt"
    data = pd.read_csv(file, sep=',')

    seqs = data['seqs'].values
    labels = data['labels'].values

    seqs = seqs.tolist()
    labels = labels.tolist()

    X, y = sample_seqs(seqs, labels)

    #print(X[0])
    #print(y[0])

    assert X[0] == 'ACATCCGTGCACCTCCG'
    assert y[0] == 1

test_sample_seqs()


def test_one_hot_encode_seqs():
    from nn.preprocess import one_hot_encode_seqs
    import numpy as np

    # L:oading the data test_seqs.txt
    file = "/Users/cindybarrios/Desktop/BMI203_algorithms/final-nn-cb/data/test_data.txt"
    data = pd.read_csv(file, sep=',')

    seqs = data['seqs'].values
    labels = data['labels'].values

    seqs = seqs.tolist()
    labels = labels.tolist()

    X, y = sample_seqs(seqs, labels)

    seqs = one_hot_encode_seqs(X) # One-hot encode the sequences
    labels = np.array(y, dtype=int) # Convert the labels to a numpy array

    #print(seqs[0][0])
    #print(labels[0])

    assert seqs[0][0] == 1
    assert labels[0] == 1

test_one_hot_encode_seqs()