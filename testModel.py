from nn import SimpleFFNN
import os 
import numpy as np
class TestModel:
    
    def __init__(self,filename) -> None:
        file="data/"+filename+".pkl"
        if os.path.isfile(file):
            self.nn= SimpleFFNN.load_model(file)
        else:
            print(f"File '{file}'not found")

    def test_with_label(self, data_to_test):
        print("\033[34mTesting the Model with Labels\n\033[0m")
        
        genres = ['drama', 'comedy', 'horror', 'action', 'romance', 'western', 'animation', 'crime', 'sci-fi']
        
        for test in data_to_test:
            features = np.array([test[0], test[1], test[3], test[4]])
            
            # Passar as features para o forward
            output = self.nn.forward(features.reshape(1, -1))  # reshaping para passar como 2D array
            
            # Encontrar o índice do gênero previsto
            predicted_genre_index = np.argmax(output)
            
            # Obter o gênero previsto usando o índice
            predicted_genre = genres[predicted_genre_index]
            
            # Comparar com o gênero real (coluna 2)
            actual_genre = test[2]
            
            if predicted_genre == actual_genre:
                print(f"\033[34m'Model: {predicted_genre}', Test: '{actual_genre}'\033[0m")
            else:
                print(f"\033[31m'Model: {predicted_genre}', Test: '{actual_genre}'\033[0m")
        
        print("\033[36mTest Completed!\n\033[0m")

        
    def test_without_labels(self,data_to_test):
        print("\033[34m Testing the Model without Labels\n\033[0m")
        
        genres = ['drama', 'comedy', 'horror', 'action', 'romance', 'western', 'animation', 'crime', 'sci-fi']
        results=[]
        for test in data_to_test: 
            output = self.nn.forward(test)
            predicted_genre_index = self.nn.argmax(output)
            results.append(genres[predicted_genre_index])

        with open("results.txt", 'w') as file:
            for row in results:
                file.write(row + "\n")
                
        print("\033[36mTest Completead!\n\033[0m")