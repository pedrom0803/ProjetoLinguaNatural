from nn import SimpleFFNN
import os 
class TestModel:
    
    def __init__(self,filename) -> None:
        file="data/"+filename+".pkl"
        if os.path.isfile(file):
            self.nn= SimpleFFNN.load_model(file)
        else:
            print(f"File '{file}'not found")

    def test_with_label(self,data_to_test):        
        print("\033[34m Testing the Model with Labels\n\033[0m")
        
        genres = ['drama', 'comedy', 'horror', 'action', 'romance', 'western', 'animation', 'crime', 'sci-fi']
        for test in data_to_test: 
            output = self.nn.forward(test[:, [0, 1, 3, 4]])
            predicted_genre_index = self.nn.argmax(output)
            # O gênero previsto é o que corresponde ao índice com maior probabilidade
            predicted_genre = genres[predicted_genre_index]
            if predicted_genre==test[:, 2]:
                print(f"\033[34m'Model: {predicted_genre}',Test: '{test[:, 2]}'")
            else:
                print(f"\033[31m'Model: {predicted_genre}',Test: '{test[:, 2]}'")

        print("\033[36mTest Completead!\n\033[0m")
        
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