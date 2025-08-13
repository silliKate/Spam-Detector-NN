import pandas as pd
import numpy as np
import re
import copy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


# used to calcuate A during forward propagation
def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z): return np.maximum(0, z)


# used to calculate dZ during backpropagation
def sigmoid_backward(dA, Z): return dA * (sigmoid(Z) * (1 - sigmoid(Z)))
def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True) # just copy dA
    dZ[Z < 0] = 0 # zero out where Z < 0
    return dZ


def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    return text


def load_data(file_path, test_size = 0.1):
    # load the dataset
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    # renaming columns
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    
    # map labels to binary
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # preprocess the text
    df['text'] = df['text'].apply(preprocess_text)

    # vectorize text
    vectorizer = TfidfVectorizer(max_features = 7000)
    X = vectorizer.fit_transform(df['text']).toarray()
    Y = df['label'].values.reshape(-1, 1)
    
    # Split into train and dev
    X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size = test_size, random_state = 42, stratify=Y)
    
    return X_train.T, X_dev.T, Y_train.T, Y_dev.T, vectorizer


def initialize_params(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def linear_forward(A_prev, W, b):
    # A_prev is the activation from the previous layer
    # W is the weight matrix
    # b is the bias vector

    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    # A_prev is the activation from the previous layer
    # W is the weight matrix
    # b is the bias vector
    # activation is either "sigmoid" or "relu"

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = relu(Z)
    else:
        raise ValueError("Unsupported activation function: {}".format(activation))
    
    cache = (linear_cache, Z) #((A_prev, W, b), Z)

    return A, cache


def forward_propagation(X, parameters):
    # X is the input data with shape (features, number of examples)
    # parameters is the output of initialize_params

    caches = [] # list to store caches for each layer, [((A0, W1, b1), Z1), ((A1, W2, b2), Z2), ...]
    A = X
    L = len(parameters) // 2 # '// 2' since we have both W and b

    # Linear -> Relu for L-1 layers
    for l in range(1, L):
        A_prev = A

        # we send previous activation function A_prev, to get the next activation A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)

    # Linear -> Sigmoid for the last layer
    # AL is the activation of the last layer, aka the prediction of the model
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = 'sigmoid')
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]

    logprobs = np.dot(Y, np.log(AL).T) + np.dot(1-Y, np.log(1-AL).T)
    cost = - np.sum(logprobs) / m
    
    cost = np.squeeze(cost) # converts cost to a scalar

    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache # cache is the output of linear_forward, a tuple containing (A_prev, W, b)
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis = 1, keepdims = True) / m
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, Z = cache # cache is the output of linear_activation_forward, a tuple containing ((A_prev, W, b), Z)
    
    if activation == "relu":
        dZ = relu_backward(dA, Z)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, Z)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    else:
        raise ValueError("Unsupported activation function: {}".format(activation))

    return dA_prev, dW, db

def backward_propagation(AL, Y, caches):
    # caches is a list of caches from forward propagation, in the form [((A0, W1, b1), Z1), ((A1, W2, b2), Z2), ...]
    # caches[-1] is the cache for the last layer
    # caches[l] is the cache for the lth layer

    grads = {} # dictionary to hold gradients for each layer
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # ensure Y has the same shape as AL
    
    dAL = - np.divide(Y,AL) + np.divide(1-Y,1-AL) # calculating the dA for the last layer
    
    # Sigmoid -> Linear for the last layer
    current_cache = caches[-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    
    # Relu -> Linear for L-1 layers
    # loops from l = L-2 to l = 0
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(params, grads, learning_rate):
    # updates parameters using gradient descent
    
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2 # '// 2' since we have both W and b

    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]

    return parameters


def nn_model(X, Y, layers_dims, learning_rate = 0.01, num_iterations = 3000, print_cost = False):
    # X is the input data, of shape (features, number of examples)
    # Y is the  true "label" vector, 0 if "ham", 1 if "spam", of shape (1, number of examples)
    # layers_dims is the list containing the input size and each layer size, of length (number of layers + 1).

    costs = [] # a list to keep track of cost
    
    parameters = initialize_params(layers_dims) # initialize
    for i in range(0, num_iterations): # loop, with gradient descent algorithm
        AL, caches = forward_propagation(X, parameters) # forward propagation 
        cost = compute_cost(AL, Y) # compute cost
        grads = backward_propagation(AL, Y, caches) # backward propagation
        parameters = update_parameters(parameters, grads, learning_rate) # update parameters

        # prints the cost every 100 iterations and for the last iteration
        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0:
            costs.append(cost)
    
    return parameters, costs


def predict(X, Y, parameters):
    # X is the input data, of shape (features, number of examples)
    # parameters is the output of nn_model

    AL, _ = forward_propagation(X, parameters) # forward propagation
    predictions = (AL > 0.5).astype(int) # convert probabilities to binary predictions
    accuracy = np.mean(predictions == Y) * 100
    return accuracy


# Main script to train the neural network on the spam dataset
def main():
    parameters, costs = nn_model(X_train, Y_train, layers_dims, learning_rate, num_iterations, print_cost = True)

    # Save the parameters and vectorizer in a .npz file
    np.savez("data/model_params.npz", parameters = parameters, vectorizer = V)
    print("Parameters and Vectorizer for the data has been stored in data/model_params.npz")

    train_acc = predict(X_train, Y_train, parameters)
    dev_acc = predict(X_dev, Y_dev, parameters)

    print(f"Train accuracy: {train_acc:.2f}%")
    print(f"Dev accuracy: {dev_acc:.2f}%")

    return parameters, costs


X_train, X_dev, Y_train, Y_dev, V = load_data('data/spam.csv', test_size = 0.1)

# hyperparameters
learning_rate = 0.01
num_iterations = 3000
nx = X_train.shape[0]  # number of input features
n1 = 20 # first hidden layer size
n2 = 10 # second hidden layer size
n3 = 5 # third hidden layer size
ny = 1  # output layer size
layers_dims = [nx, n1, n2, n3, ny]

# main()


# <-------------------------------------------------------------------------------------------> #
# for plotting & testing purposes
def dev(X, Y, learning_rate = 0.01, num_iterations = 3000, layers_dims = [7000, 20, 10, 5, 1]):
    parameters, costs = nn_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost = True)
    return parameters, costs


# learning rates vs costs
def plot_learning_rates(lrs, num_iterations=3000, layers_dims=[7000, 20, 10, 5, 1]):
    results = {}
    
    for lr in lrs:
        print(f"Testing learning rate: {lr}")
        _, costs = dev(X_dev, Y_dev, learning_rate=lr, num_iterations=num_iterations, layers_dims=layers_dims)
        results[lr] = costs

    plt.figure(figsize=(8, 6))
    for lr, costs in results.items():
        plt.plot(range(len(costs)), costs, label=f"lr={lr}")

    plt.ylabel('Cost')
    plt.xlabel('Iterations (per hundreds)')
    plt.title("Learning Rate Comparison (for training set)")
    plt.legend()
    plt.grid(True)
    plt.show()

# plot_learning_rates([0.001, 0.005, 0.0075, 0.01, 0.1, 0.5])


# Plotting the cost over iterations
def plot_costs(learning_rate):
    _, costs = dev(X_train, Y_train, learning_rate, num_iterations = 3000, layers_dims=[X_train.shape[0], 20, 10, 5, 1])
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per hundreds)')
    plt.title(f"Learning rate = {learning_rate}")
    plt.savefig("plot1.png")
    plt.show()

# plot_costs(learning_rate = 0.01)


# ROC
def plot_roc_curve(X, Y):
    parameters, _ = nn_model(X, Y, layers_dims = [7000, 20, 10, 5, 1], learning_rate = 0.01, num_iterations = 3000, print_cost = True)
    AL, _ = forward_propagation(X, parameters)
    AL = np.squeeze(AL)
    fpr, tpr, _ = roc_curve(Y.flatten(), AL)
    roc_auc = auc(fpr, tpr)
    
    print(f"AUC: {roc_auc:.2f}")
    plt.figure()
    plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.show()

plot_roc_curve(X_train, Y_train)

# <-------------------------------------------------------------------------------------------> #
