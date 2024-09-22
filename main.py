import os
from nn import SimpleFFNN
from train import Train
from preProcessing import PreProcessing
import numpy as np


def split_array(data: np.ndarray, train_size: float = 0.75):
    """
    Divide um array 2D em dois arrays: um com train_size dos dados e outro com o restante.
    
    :param data: O array 2D a ser dividido.
    :param train_size: A proporção de dados a serem usados para o primeiro array (default é 0.75).
    :return: Dois arrays 2D, um com os dados de treinamento e outro com os dados de teste.
    """
    # Calcula o índice para a divisão
    split_index = int(len(data) * train_size)
    
    # Embaralha os dados
    np.random.shuffle(data)
    
    # Divide o array
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    return train_data, test_data


if __name__ == "__main__":
    
    genres = ['drama', 'comedy', 'horror', 'action', 'romance', 'western', 'animation', 'crime', 'sci-fi']
    
    #file onde já está o modelo treinado
    nnFile="ficheiro_com_modelo_treinado"
    model=None

    nodes_input=4
    nodes_outuput=len(genres)

    #limpesa de ficheiro de treino
    data="train"
    
    print("\033[34mPre Processing the data\n\033[0m")
    
    pp=PreProcessing(data+".txt")
    clean_data=pp.returnCleanText()
    
    print("\033[32mPre Processing Completed!\n\033[0m")
    

    # Dividir os dados em treino e teste
    data_to_train, data_to_test = split_array(clean_data)

    # Carregar o modelo se ele já existir
    if os.path.isfile(nnFile):
        print("\033[34mLoading Model\n\033[0m")
        
        model=SimpleFFNN.load_model(nnFile+".pkl")
        
        print("\033[32mLoading Completed!\n\033[0m")
    # Se não existir, cria e treina um novo modelo   
    else:
        print("\033[34mCreating a new Model\n\033[0m")
        
        newNNFile="ficheiro_com_novo_modelo"
        layer_sizes = [nodes_input, 5, nodes_outuput] # adicionar o numero de nos por camada,hidden, que bem se entender
        learning_rate = 0.01
        epochs=100
        model = Train(data_to_train,newNNFile.join(".pkl"),layer_sizes, learning_rate, epochs)
        model.train()
        
        print("\033[32mModel Created!\n\033[0m")
        
        
    print("\033[34m Testing the Model\n\033[0m")
    
    for test in data_to_test: 
        output = model.forward(test[:, [0, 1, 3, 4]])
        predicted_genre_index = np.argmax(output)
        # O gênero previsto é o que corresponde ao índice com maior probabilidade
        predicted_genre = genres[predicted_genre_index]
        if predicted_genre==test[:, 2]:
            print(f"\033[34m'Model: {predicted_genre}',Test: '{test[:, 2]}'")
        else:
            print(f"\033[31m'Model: {predicted_genre}',Test: '{test[:, 2]}'")

    print("\033[36mTest Completead!\n\033[0m")




