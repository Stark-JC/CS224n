import numpy as np
import matplotlib.pyplot as plt

N = 100  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes
Ntrain = N * K

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


class Model(object):
    def __init__(self, h1, h2):
        self.model = {}
        self.model['h1'] = h1  # size of hidden layer 1
        self.model['h2'] = h2  # size of hidden layer 2
        self.model['W1'] = 0.1 * np.random.randn(D, h1)
        self.model['b1'] = np.zeros((1, h1))
        self.model['W2'] = 0.1 * np.random.randn(h1, h2)
        self.model['b2'] = np.zeros((1, h2))
        self.model['W3'] = 0.1 * np.random.randn(h2, K)
        self.model['b3'] = np.zeros((1, K))


def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x


def sigmoid_grad(x):
    return (x) * (1 - x)


def relu(x):
    return np.maximum(0, x)


def generate():
    np.random.seed(0)
    X = np.zeros((N * K, D))
    y = np.zeros(N * K, dtype='uint8')
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.cos(t), r * np.sin(t)]  # 列拼接
        y[ix] = j
    return X, y


def plot_class(X, y):
    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()


def three_layer_net(NONLINEARITY, X, y, model, step_size=0.001, reg=0.5, epochs=10000):
    # X: N * D
    # y: N * 1
    # parameter initialization

    h1 = model['h1']
    h2 = model['h2']
    W1 = model['W1']  # D * h1
    W2 = model['W2']  # h1 * h2
    W3 = model['W3']  # h2 * K
    b1 = model['b1']  # h1 * 1
    b2 = model['b2']  # h2 * 1
    b3 = model['b3']  # K * 1

    # some hyperparameters

    # gradient descent loop
    num_examples = X.shape[0]
    plot_array_1 = []
    plot_array_2 = []
    for i in range(epochs):

        # FOWARD PROP

        if NONLINEARITY == 'RELU':
            hidden_layer = relu(np.dot(X, W1) + b1)
            hidden_layer2 = relu(np.dot(hidden_layer, W2) + b2)
            scores = np.dot(hidden_layer2, W3) + b3

        elif NONLINEARITY == 'SIGM':
            hidden_layer = sigmoid(np.dot(X, W1) + b1)
            hidden_layer2 = sigmoid(np.dot(hidden_layer, W2) + b2)
            scores = np.dot(hidden_layer2, W3) + b3

        # softmax
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]

        # compute the loss: average cross-entropy loss and regularization
        corect_logprobs = -np.log(probs[range(num_examples), y])  # nice code
        data_loss = np.sum(corect_logprobs) / num_examples
        reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2) + 0.5 * reg * np.sum(W3 * W3)
        loss = data_loss + reg_loss
        if i % 1000 == 0:
            print("iteration %d: loss %f" % (i, loss))

        # compute the gradient on scores
        dscores = probs  # N x K
        dscores[range(num_examples), y] -= 1  # nice code，每一行表示一个样本的输出层各个单元的误差
        dscores /= num_examples  # 后面算sum的时候不用除数量了

        # BACKPROP HERE
        dW3 = (hidden_layer2.T).dot(dscores)
        db3 = np.sum(dscores, axis=0, keepdims=True)

        if NONLINEARITY == 'RELU':

            # backprop ReLU nonlinearity here
            dhidden2 = np.dot(dscores, W3.T)  # N x h2
            dhidden2[hidden_layer2 <= 0] = 0
            dW2 = np.dot(hidden_layer.T, dhidden2)
            plot_array_2.append(np.sum(np.abs(dW2)) / np.sum(np.abs(dW2.shape)))
            db2 = np.sum(dhidden2, axis=0)
            dhidden = np.dot(dhidden2, W2.T)
            dhidden[hidden_layer <= 0] = 0

        elif NONLINEARITY == 'SIGM':

            # backprop sigmoid nonlinearity here
            dhidden2 = dscores.dot(W3.T) * sigmoid_grad(hidden_layer2)  # N x h2
            dW2 = (hidden_layer.T).dot(dhidden2)
            plot_array_2.append(np.sum(np.abs(dW2)) / np.sum(np.abs(dW2.shape)))
            db2 = np.sum(dhidden2, axis=0)
            dhidden = dhidden2.dot(W2.T) * sigmoid_grad(hidden_layer)

        dW1 = np.dot(X.T, dhidden)
        plot_array_1.append(np.sum(np.abs(dW1)) / np.sum(np.abs(dW1.shape)))
        db1 = np.sum(dhidden, axis=0)

        # add regularization
        dW3 += reg * W3
        dW2 += reg * W2
        dW1 += reg * W1

        # option to return loss, grads -- uncomment next comment
        grads = {}
        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['W3'] = dW3
        grads['b1'] = db1
        grads['b2'] = db2
        grads['b3'] = db3
        # return loss, grads

        # update
        W1 += -step_size * dW1
        b1 += -step_size * db1
        W2 += -step_size * dW2
        b2 += -step_size * db2
        W3 += -step_size * dW3
        b3 += -step_size * db3
    # evaluate training set accuracy
    if NONLINEARITY == 'RELU':
        hidden_layer = relu(np.dot(X, W1) + b1)
        hidden_layer2 = relu(np.dot(hidden_layer, W2) + b2)
    elif NONLINEARITY == 'SIGM':
        hidden_layer = sigmoid(np.dot(X, W1) + b1)
        hidden_layer2 = sigmoid(np.dot(hidden_layer, W2) + b2)
    scores = np.dot(hidden_layer2, W3) + b3
    predicted_class = np.argmax(scores, axis=1)
    print('training accuracy: %.2f' % (np.mean(predicted_class == y)))
    # return cost, grads
    return plot_array_1, plot_array_2, W1, W2, W3, b1, b2, b3


if __name__ == "__main__":
    X, y = generate()
    # plot_class(X, y)
    model = Model(50, 50).model
    (sigm_array_1, sigm_array_2, s_W1, s_W2, s_W3, s_b1, s_b2, s_b3) = three_layer_net('SIGM', X, y, model,
                                                                                       step_size=1e-1, reg=1e-3)
