import numpy as np
import pickle

class SimpleFFNN:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.vectorizer=None
        self.encoder=None
        # Inicializa pesos e viés para cada camada
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]))
            self.biases.append(np.random.randn(layer_sizes[i + 1]))
    
    @classmethod
    def from_weights_bias(cls, weights, bias, vectorizer, encoder):
        instance = cls([])  # Cria uma instância sem camada
        instance.weights = weights
        instance.biases = bias
        instance.vectorizer=vectorizer
        instance.encoder=encoder
        return instance
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # Subtração para estabilidade numérica
        return exp_x / np.sum(exp_x, axis=0)

    def forward(self, x):
        self.activations = [x]
        # Compute activations for each layer
        for i in range(len(self.weights)):
            x = self.sigmoid(np.dot(x, self.weights[i]) + self.biases[i])
            self.activations.append(x)
        # Para a última camada, use softmax
        # x = self.softmax(np.dot(x, self.weights[-1]) + self.biases[-1])
        # self.activations.append(x)
        
        return x

    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(x.shape[0]):
                # Forward pass
                output = self.forward(x[i])

                # Calculate error
                errors = [y[i] - output]

                # Calculate errors for each layer
                for l in range(len(self.weights) - 1, 0, -1):
                    error = errors[0].dot(self.weights[l].T)
                    errors.insert(0, error)

                # Update weights and biases from output to input layer
                for j in range(len(self.weights)):
                    d_activations = errors[j] * self.sigmoid_derivative(self.activations[j + 1])
                    self.weights[j] += np.outer(self.activations[j], d_activations) * learning_rate
                    self.biases[j] += d_activations * learning_rate

            if epoch % 20 == 0:
                loss = np.mean(np.square(y - self.forward(x)))
                print(f'Epoch {epoch}, Loss {loss}')
                
    #Guardar o modelo depois de o treinar
    def save_model(self, filename,vectorizer,encoder):
        with open(f'data/{filename}.pkl', 'wb') as f:
            pickle.dump({'weights': self.weights, 'biases': self.biases,'vectorizer': vectorizer, 'encoder':encoder}, f)
            
    #Usar o modelo para o poder testar
    def load_model(filename):
        with open(f'{filename}', 'rb') as f:
            data = pickle.load(f)
            return SimpleFFNN.from_weights_bias(data['weights'],data['biases'],data['vectorizer'],data['encoder'])