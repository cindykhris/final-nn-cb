# Imports
import numpy as np
from typing import List, Dict, Tuple, Union

from matplotlib import pyplot as plt
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.
    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.
    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(self, nn_arch: List[Dict[str, Union[int, str]]],
                 lr: float, seed: int, batch_size: int, epochs: int, loss_function: str):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!
        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.
        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict
    
    def _single_forward(self, W_curr: ArrayLike, b_curr: ArrayLike,
                        A_prev: ArrayLike, activation: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.
        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.
        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """

        # Set current layer Z matrix to be the weights matrix multiplied by the previous layer's activation matrix plus the bias matrix
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        # Set current layer activation matrix to be the activation function of the current layer Z matrix
        # If the activation function is relu, use the relu function 
        if activation == 'relu':
            A_curr = self._relu(Z_curr)
        # If the activation function is sigmoid, use the sigmoid function
        elif activation == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)
        # If neither relu nor sigmoid, raise an exception
        else:
            raise Exception('Non-supported activation function. Please specify either "relu" or "sigmoid" as activation functions.')
        
        return A_curr, Z_curr # return Tuple[ArrayLike, ArrayLike] of current layer activation matrix and current layer linear transformed matrix
    
    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.
        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].
        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """

        # Initialize cache dictionary
        cache = {'A0': X}

        # Set current layer's activation matrix to be the input matrix
        A_curr = X

        # Loop through each layer in the neural network
        for idx, layer in enumerate(self.arch):
            
            # Set current layer's activation matrix to be the previous layer's activation matrix
            A_prev = A_curr

            # Get index of first layer (layer 1)
            layer_idx = idx + 1
            
            # Extract corresponding weight matrix and bias matrix for current layer from parameter dictionary
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            activation = layer['activation']

            # Perform a single forward pass on the current layer
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation)

            # Store current layer's Z matrix and A matrix in cache
            cache['Z' + str(layer_idx)] = Z_curr
            cache['A' + str(layer_idx)] = A_curr

        # Set output to be the final layer's activation matrix
        output = A_curr.T

        return output, cache
    
    def _single_backprop(self, W_curr: ArrayLike, b_curr: ArrayLike, Z_curr: ArrayLike, A_prev: ArrayLike,
                         dA_curr: ArrayLike, activation_curr: str) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.
        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.
        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """

        # If the activation function is relu, use the relu derivative function
        if activation_curr == 'relu':
            activation_func = self._relu_backprop(dA_curr, Z_curr)

        # If the activation function is sigmoid, use the sigmoid derivative function
        elif activation_curr == 'sigmoid':
            activation_func = self._sigmoid_backprop(dA_curr, Z_curr)

        # If neither relu nor sigmoid, raise an exception
        else:
            raise Exception('Non-supported activation function. Please specify either "relu" or "sigmoid" as activation functions.')
        
        # Set current layer's partial derivative of loss function with respect to current layer's Z matrix to be the activation function derivative multiplied by the current layer's partial derivative of loss function with respect to current layer's activation matrix
        dZ_curr = activation_func

        # Compute the gradient for the current layer
        dW_curr = np.dot(dZ_curr, A_prev.T) / np.shape(A_prev)[1]
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / np.shape(A_prev)[1]
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr # return Tuple[ArrayLike, ArrayLike, ArrayLike] of partial derivative of loss function with respect to previous layer activation matrix, partial derivative of loss function with respect to current layer weight matrix, and partial derivative of loss function with respect to current layer bias matrix
    
    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.
        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.
        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """

        # Initialize gradient dictionary
        grad_dict = {}

        # Set the partial derivative of loss function with respect to final layer activation matrix to be the difference between the predicted output and the ground truth labels
        if self._loss_func == 'mse':
            dA_prev = self._mean_squared_error_backprop(y, y_hat)
        elif self._loss_func == 'bce':
            dA_prev = self._binary_cross_error_backprop(y, y_hat)
        else:
            raise Exception('Non-supported loss function. Please specify either "mse" or "bce" as loss functions.')

        # Loop through each layer in the neural network
        for layer_idx_prev, layer in reversed(list(enumerate(self.arch))):

            # Get index of current layer
            layer_idx_curr = layer_idx_prev + 1
            dA_curr = dA_prev

            # Extract corresponding activation matrix and Z matrix for current layer from cache
            if layer_idx_prev == 0:
                A_prev = cache['A0']
            else:
                A_prev = cache['A' + str(layer_idx_prev)]

            # Extract corresponding weight matrix and bias matrix for current layer from parameter dictionary
            W_curr = self._param_dict['W' + str(layer_idx_curr)]
            b_curr = self._param_dict['b' + str(layer_idx_curr)]
            Z_curr = cache['Z' + str(layer_idx_curr)]

            # Extract activation function for current layer
            activation_curr = layer['activation']

            # Perform a single backprop pass on the current layer
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_prev, activation_curr)

            # Store current layer's partial derivative of loss function with respect to current layer's weight matrix, current layer's partial derivative of loss function with respect to current layer's bias matrix, and previous layer's partial derivative of loss function with respect to previous layer's activation matrix in gradient dictionary
            grad_dict['dW' + str(layer_idx_curr)] = dW_curr
            grad_dict['db' + str(layer_idx_curr)] = db_curr
            grad_dict['dA' + str(layer_idx_prev)] = dA_prev

        return grad_dict
    
    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything
        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """

        # Loop through each layer in the neural network and update the parameters for W and b
        for layer_idx, layer in enumerate(self.arch, 1):
            self._param_dict['W' + str(layer_idx)] -= self._lr * grad_dict['dW' + str(layer_idx)]
            self._param_dict['b' + str(layer_idx)] -= self._lr * grad_dict['db' + str(layer_idx)]

    def fit(self, X_train: ArrayLike, y_train: ArrayLike,
            X_val: ArrayLike, y_val: ArrayLike) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.
        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.
        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """

        # Initialize lists of training and validation losses
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        # Iterate across each epoch
        for epoch in range(self._epochs):
            # Shuffle indices to randomly select batches
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)

            # Initialize list to keep track of different batches
            batches = []

            # Iterate through rows of X_train in increments of batch size
            for i in range(0, X_train.shape[0], self._batch_size):
                X_batch = X_train[i:i + self._batch_size]  # Use batch indices to select corresponding rows from X_train
                y_batch = y_train[i:i + self._batch_size]  # Use batch indices to select corresponding rows from y_train
                batches.append((X_batch, y_batch))  # Add the batch to the list of batches

            # Iterate through the X and y batches
            for X_batch, y_batch in batches:
                X_batch = X_batch.T  # Transpose each X batch
                y_batch = y_batch.T  # Transpose each X batch
                y_hat, cache = self.forward(X_batch)  # Forward pass on X batches
                grad_dict = self.backprop(y_batch, y_hat, cache)  # Backpropagate using y_batches
                self._update_params(grad_dict)  # Update parameters using values stored from backpropagation

            # Calculate predictions from training and validation sets
            y_hat_train = self.predict(X_train)
            y_hat_val = self.predict(X_val)

            # Calculate training and validation losses using user-defined loss function
            if "bce" in self._loss_func:
                train_loss = self._binary_cross_error(y_train.T, y_hat_train.T)
                val_loss = self._binary_cross_error(y_val.T, y_hat_val.T)
            elif "mse" in self._loss_func:
                train_loss = self._mean_squared_error(y_train.T, y_hat_train.T)
                val_loss = self._mean_squared_error(y_val.T, y_hat_val.T)
            else:
                raise NameError("Loss function name is not defined. "
                                "Choose either mean squared error or binary cross entropy as loss function.")

            # Append calculated loss to per epoch loss lists
            per_epoch_loss_train.append(train_loss)
            per_epoch_loss_val.append(val_loss)

        return per_epoch_loss_train, per_epoch_loss_val
                
    
    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function makes predictions for the input features.
        Args:
            X: ArrayLike
                Input features.
        Returns:
            y_hat: ArrayLike
                Predicted output values.
        """
    
        # Perform forward pass
        y_hat, _ = self.forward(X.T)
    
        return y_hat
    
    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.
        Args:
            Z: ArrayLike
                Output of layer linear transform.
        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """

        nl_transform = 1 / (1 + np.exp(-Z))
    
        return nl_transform
    
    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.
        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.
        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """

        sig = self._sigmoid(Z)
        dZ = dA * sig * (1 - sig)
    
        return dZ
    
    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.
        Args:
            Z: ArrayLike
                Output of layer linear transform.
        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(0, Z)
    
    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.
        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.
        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """

        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
    
        return dZ
    
    def _binary_cross_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.
        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.
        Returns:
            loss: float
                Average loss over mini-batch.
        """

        loss = -(1 / y.shape[1] * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))
    
        return loss
    
    def _binary_cross_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.
        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.
        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        dA = - (np.divide(y, y_hat.T) - np.divide((1 - y), (1 - y_hat.T)))
        return dA
    
    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.
        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.
        Returns:
            loss: float
                Average loss of mini-batch.
        """

        loss = (1 / (2 * y.shape[1]) * np.sum(np.square(y_hat - y)))
    
        return loss
    
    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.
        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.
        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # Derivative of the MSE loss function
        return (2 / y.shape[1]) * (y_hat.T - y)