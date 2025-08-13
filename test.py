import numpy as np

# activation functions
def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z): return np.maximum(0, z)


def forward_propagation(X, parameters):
    A = X
    L = len(parameters) // 2

    # hidden layers with relu activation
    for l in range(1, L):
        W = parameters[f"W{l}"]
        b = parameters[f"b{l}"]
        A = relu(np.dot(W, A) + b)

    # output layer with sigmoid activation
    W = parameters[f"W{L}"]
    b = parameters[f"b{L}"]
    AL = sigmoid(np.dot(W, A) + b)

    return AL


def predict(X, parameters):
    # returns 1 for spam, 0 for not spam

    AL = forward_propagation(X, parameters)
    AL = float(np.squeeze(AL))  # converts AL to a scalar
    return AL > 0.5, AL # prediction, probability


# load model parameters and vectorizer
data = np.load("data/model_params.npz", allow_pickle=True)
parameters = data["parameters"].item()
V = data["vectorizer"].item()

# main loop for user input
while True:
    text = input("Enter text (0 to exit): ")
    if text == "0":
        break

    X = V.transform([text]).toarray().T
    label, probs = predict(X, parameters)
    probs = round(probs * 100, 1)  # convert to percentage
    print(f"Spam ({probs}% confident)" if label else f"Not Spam ({100 - probs}% confident)")
