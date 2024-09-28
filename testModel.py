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

    def test_with_label(self, data_to_test,genre_index=2,plot_index=4):
        print("\033[34mTesting the Model with Labels\n\033[0m")
        
        total = 0.0
        corrects = 0.0
        

        for test in data_to_test:
            total += 1.0
            
            plot_vector = self.nn.vectorizer.transform([test[int(plot_index)]]).toarray()  # Vetorizar o texto
            prediction = self.nn.forward(plot_vector)  # Fazer a predição
            predicted_genre = self.nn.encoder.inverse_transform(prediction)  # Decodificar o gênero
            
            if predicted_genre == test[int(genre_index)]:
                print(f"\033[32mFor the movie '{test[0]}' the model said: {predicted_genre}', and was: '{test[int(genre_index)]}'\033[0m")
                corrects += 1.0
            else:
                print(f"\033[31mFor the movie '{test[0]}' the model said: {predicted_genre}', and was: '{test[int(genre_index)]}'\033[0m")
        
        success = float(corrects / total) * 100
        
        print("\033[34mTest Completed with ", f"\033[36m{success}% ", "\033[34mof success\033[0m")


        
    def test_without_labels(self, data_to_test):
        print("\033[34mTesting the Model without Labels\n\033[0m")
        
        results = []

        for test in data_to_test:
            plot_vector = self.nn.vectorizer.transform([test[3]]).toarray()  # Vetorizar o texto
            prediction = self.nn.forward(plot_vector)  # Fazer a predição
            predicted_genre = self.nn.encoder.inverse_transform(prediction)  # Decodificar o gênero
            results.append(predicted_genre)

        # Escrever os resultados no arquivo "results.txt"
        with open("data/results.txt", 'w') as file:
            for predict in results:
                file.write(predict + "\n")
        
        print("\033[36mTest Completed!\n\033[0m")
        
    def test_from_input(self,text,genre):
        
        print("\033[34mTrying to predict the genre\n\033[0m")
        plot_vector = self.nn.vectorizer.transform([text]).toarray()  # Vetorizar o texto
        prediction = self.nn.forward(plot_vector)  # Fazer a predição
        predicted_genre = self.nn.encoder.inverse_transform(prediction)  # Decodificar o gênero
        
        if predicted_genre == genre:
            print(f"\033[32mFor the movie given by input the model said: {predicted_genre}', and was: '{genre}'\033[0m")
        else:
            print(f"\033[31mFor the movie given by input the model said: {predicted_genre}', and was: '{genre}'\033[0m")
        
