from nn import SimpleFFNN
import os 
import numpy as np

class TestModel:
    
    def __init__(self, filename) -> None:
        file = "data/" + filename + ".pkl"
        if os.path.isfile(file):
            self.nn = SimpleFFNN.load_model(file)  # Carregar o modelo
        else:
            print(f"File '{file}' not found")

    def test_with_label(self, data_to_test):
        print("\033[34mTesting the Model with Labels\n\033[0m")
        
        genres = ['drama', 'comedy', 'horror', 'action', 'romance', 'western', 'animation', 'crime', 'sci-fi']
        
        total = 0.0
        corrects = 0.0
        

        for test in data_to_test:
            total += 1.0
            
            plot_vector = self.nn.vectorizer.transform([test[4]]).toarray()  # Vetorizar o texto
            prediction = self.nn.forward(plot_vector)  # Fazer a predição
            predicted_genre = self.nn.encoder.inverse_transform(prediction)  # Decodificar o gênero
            
            if predicted_genre == test[2]:
                print(f"\033[34m'Model: {predicted_genre}', Test: '{test[2]}'\033[0m")
                corrects += 1.0
            else:
                print(f"\033[31m'Model: {predicted_genre}', Test: '{test[2]}'\033[0m")
        
        success = float(corrects / total) * 100
        
        print("\033[36mTest Completed!\033[0m", " with ", f"\033[34m{success}% of success\033[0m")


        
    def test_without_labels(self, data_to_test):
        print("\033[34mTesting the Model without Labels\n\033[0m")
        
        genres = ['drama', 'comedy', 'horror', 'action', 'romance', 'western', 'animation', 'crime', 'sci-fi']
        results = []

        for test in data_to_test:
            # Extrair as colunas de texto
            title = [test[0]]  # Precisa estar em formato de lista para o encoder
            origin = [test[1]]
            director = [test[3]]
            plot = [test[4]]

            # Usar o vectorizer e encoder do modelo para transformar os dados de texto
            title_features = self.nn.encoder.transform(np.array(title).reshape(-1, 1)).toarray()
            origin_features = self.nn.encoder.transform(np.array(origin).reshape(-1, 1)).toarray()
            director_features = self.nn.encoder.transform(np.array(director).reshape(-1, 1)).toarray()
            plot_features = self.nn.vectorizer.transform(plot).toarray()

            # Combinar todas as features
            features = np.hstack((title_features, origin_features, director_features, plot_features))

            # Passar as features para o forward
            output = self.nn.forward(features.reshape(1, -1))  # reshaping para passar como 2D array

            # Encontrar o índice do gênero previsto
            predicted_genre_index = np.argmax(output)

            # Adicionar o gênero previsto aos resultados
            results.append(genres[predicted_genre_index])

        # Escrever os resultados no arquivo "results.txt"
        with open("data/results.txt", 'w') as file:
            for row in results:
                file.write(row + "\n")
        
        print("\033[36mTest Completed!\n\033[0m")
