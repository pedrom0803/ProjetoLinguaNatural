import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nn import SimpleFFNN


class Train:
    def __init__(self, data_to_train: list, nnFile: str, layer_hiddens, learning_rate: float, epochs: int) -> None:
        # Converte a lista em um array NumPy para permitir indexação por colunas
        self.hidden_out=layer_hiddens
        self.data_to_train = np.array(data_to_train)
        self.export_nn = nnFile
        self.learning_rate = learning_rate
        self.epochs = epochs

    def vectorize_plots(self,plots):
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(plots).toarray()  # Convertendo para array
        return X, vectorizer

    def encode_genres(self,genres):
        encoder = LabelBinarizer()
        y = encoder.fit_transform(genres)
        return y, encoder

    def train(self):
        plots = []
        genres = []
        for parts in self.data_to_train:
            if len(parts) == 5:  # Certifica-se que a linha tem todas as 5 partes
                plot = parts[4]  # O plot é o quinto elemento (índice 4)
                plots.append(plot)
                genre = parts[2]
                genres.append(genre)
        
        # Vetorizar os plots
        x, vectorizer = self.vectorize_plots(plots)
        # Codificar os gêneros
        y, encoder = self.encode_genres(genres)
        
        # Dimensões da rede neural
        input_size = x.shape[1]  # Número de características após a vetorização

        output_size = y.shape[1]  # Número de classes de gênero (codificação one-hot)
        
        total_layer=[]
        total_layer.append(input_size)
        for hidden in self.hidden_out:
            total_layer.append(hidden)
        total_layer.append(output_size)
        model = SimpleFFNN(layer_sizes=total_layer)
        model.train(x, y, epochs=self.epochs, learning_rate=self.learning_rate)
        model.save_model(self.export_nn,vectorizer,encoder)
        