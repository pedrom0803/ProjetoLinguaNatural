import numpy as np
from nn import SimpleFFNN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


class Train:
    def __init__(self, data_to_train: list, nnFile: str, layer_hiddens,nodes_outuput, learning_rate: float, epochs: int) -> None:
        # Converte a lista em um array NumPy para permitir indexação por colunas
        self.hidden_out=layer_hiddens+nodes_outuput
        self.file = np.array(data_to_train)
        self.export_nn = nnFile
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.vectorizer = TfidfVectorizer()
        self.encoder = OneHotEncoder()

    def train(self):
        # Separar colunas
        titles = [row[0] for row in self.file]
        origins = [row[1] for row in self.file]
        genres = [row[2] for row in self.file]
        directors = [row[3] for row in self.file]
        plots = [row[4] for row in self.file]

        # Usar TF-IDF para o plot
        plot_features = self.vectorizer.fit_transform(plots).toarray()

        # Usar One-Hot Encoding para as colunas de origem, gênero e diretor
        title_features = self.encoder.fit_transform(np.array(titles).reshape(-1, 1)).toarray()
        origin_features = self.encoder.fit_transform(np.array(origins).reshape(-1, 1)).toarray()
        genre_features = self.encoder.fit_transform(np.array(genres).reshape(-1, 1)).toarray()
        director_features = self.encoder.fit_transform(np.array(directors).reshape(-1, 1)).toarray()

        # Combinar todas as features
        x = np.hstack((title_features, origin_features, director_features, plot_features))
        input_nodes=[]
        total_nodes = title_features.shape[1] + origin_features.shape[1] +  director_features.shape[1] + plot_features.shape[1]
        input_nodes.append(total_nodes)
        layer_size= input_nodes + self.hidden_out
        self.model = SimpleFFNN(layer_size)
        
        # Treinar o modelo
        self.model.train(x, genre_features, self.epochs, self.learning_rate)

        # Salvar o modelo, TF-IDF e OneHotEncoder
        self.model.save_model(self.export_nn,self.vectorizer,self.encoder)
