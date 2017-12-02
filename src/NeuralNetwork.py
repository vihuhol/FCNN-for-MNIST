import numpy as np

def logistic(x,deriv=False):
    if deriv:
        func = logistic(x)
        return func * (1 - func)
    return 1 / (1 + np.exp(-x))

def relu(x,deriv=False):
    res = np.zeros(x.shape)
    for i,row in enumerate(x):
        for j,elem in enumerate(row):
            if elem > 0:
                res[i][j] = 1
            else:
                res[i][j] = -1
    return res

def soft_max(x, deriv=False):
    if deriv:
        func = soft_max(x)
        return func * (1 - func)
    else:
        res = np.zeros(x.shape)
        for i,row in enumerate(x):
            res[i] = np.exp(row)
            s = res[i].sum()
            res[i] = res[i] / s
        return res

def euclid_error(y, y1, deriv=False):
    if deriv:
        return y - y1
    else:
        return 1/2 * sum((y - y1) ** 2, axis=0)
    return res

def cross_entropy_error(y, y1, deriv=False):
    if deriv:
        return (y + 0.0001) / (y1 + 0.0001) - (1.0001 - y) / (1.0001 - y1) 
    else:
        pass

class myMLPClassifier:
    def __init__(self, activation = 'logistic', task = 'regression', hidden_layer_sizes = (), max_iter = 60000, random_state = 0, 
        tol = 0.001, batch_size = 100):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layers_count = len(hidden_layer_sizes)
        self.max_iter = max_iter
        self.tol = tol
        self.task = task
        self.batch_size = batch_size
        if activation == 'logistic':
            self.activation = logistic
        if activation == 'relu':
            self.activation = relu
        if task == 'regression':
            self.error_func = euclid_error
            self.last_activation = self.activation
        if task == 'classification':
            self.error_func = euclid_error
            self.last_activation = logistic
        np.random.seed(random_state)

    def fit(self, X, y):
        N, Dx = X.shape;
        if self.task == 'classification':
            num_of_classes = 0
            for elem in y:
                if elem > num_of_classes - 1:
                    num_of_classes = elem + 1
            new_y = np.zeros((N, num_of_classes))
            for i in range(N):
                for j in range(num_of_classes):
                    if y[i] == j:
                        new_y[i][j] = 1
            y = new_y
        Dy = y.shape[1];

        if N < self.batch_size:
            self.batch_size = N
        if N % self.batch_size != 0:
            raise AttributeError("Batch size must divide size of data")

        self.layers = [];
        self.layers.append(np.ndarray((self.batch_size, Dx)))
        for layer_size in self.hidden_layer_sizes:
                self.layers.append(np.ndarray((self.batch_size, layer_size)))
        self.layers.append(np.ndarray((self.batch_size, Dy)))
        
        self.weights = [];
        for i in range(1, len(self.layers)):
            w = 2*np.random.random((self.layers[i-1].shape[1] , self.layers[i].shape[1])) - 1
            self.weights.append(w)

        for i in range(0, N / self.batch_size):
            print("Batch #" + str(i + 1) + ":")
            self.fit_batch(X[i:i+self.batch_size], y[i:i+self.batch_size])

    def fit_batch(self, X, y):
        for j in range(self.max_iter):
            self.layers[0] = X
            for i in range(1, len(self.layers) - 1):
                self.layers[i] = self.activation(np.dot(self.layers[i - 1], self.weights[i - 1]))
            self.layers[-1] = self.last_activation(np.dot(self.layers[-2], self.weights[-1]))
            
            error = self.error_func(y, self.layers[-1], deriv=True)
            if j % 10000 == 0:
                print "Error:" + str(np.mean(np.abs(error)))
            if np.mean(np.abs(error)) < self.tol:
                return
           
            for i in range(len(self.layers) - 2, -1, -1):
                if i + 1 == len(self.layers) - 1:
                    delta = error * self.last_activation(self.layers[i + 1], deriv=True)
                else:
                    delta = error * self.activation(self.layers[i + 1], deriv=True)
                error = delta.dot(self.weights[i].T)
                self.weights[i] += self.layers[i].T.dot(delta)

    def predict(self, X):
        self.layers[0] = X;
        for i in range(1, len(self.layers) - 1):
                self.layers[i] = self.activation(np.dot(self.layers[i - 1], self.weights[i - 1]))
        self.layers[-1] = self.last_activation(np.dot(self.layers[-2], self.weights[-1]))
        if self.task == 'classification':
            predictions = np.zeros(self.layers[-1].shape[0])
            for i,probs in enumerate(self.layers[-1]):
                max_prob = 0
                most_possible_class = 0
                for Class,prob in enumerate(probs):
                    if prob > max_prob:
                        max_prob = prob
                        most_possible_class = Class
                predictions[i] = most_possible_class
            return  predictions              
        return self.layers[-1]

    def weights_norm(self):
        norm = 0
        n = 0
        for weights_in_layer in self.weights:
            norm += weights_in_layer.sum()
            n += len(weights_in_layer)
        return norm/n

