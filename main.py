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
    #file onde já está o modelo treinado
    nnFile="ficheiro_com_modelo_treinado"
    model=None

    nodes_input=4
    nodes_outuput=9

    #limpesa de ficheiro de treino
    data="train"
    pp=PreProcessing(data.join(".txt"))
    clean_data=pp.returnCleanText()

    data_to_train, data_to_test = split_array(clean_data)

    #já existe nao treina o modelo
    if os.path.isfile(nnFile):
        model=SimpleFFNN.load_model(nnFile.join(".txt"))
    #nao existe, vai treinar um novo modelo    
    else:
        newNNFile="ficheiro_com_novo_modelo"
        layer_sizes = [nodes_input, 5, nodes_outuput] # adicionar o numero de nos por camada,hidden, que bem se entender
        learning_rate = 0.01
        epochs=100
        model = Train(data_to_train,newNNFile.join(".pkl"),layer_sizes, learning_rate, epochs)
        model.train()
        
    for test in data_to_test: 
        predicted_genre = model.forward(test[:, [0, 1, 3, 4]])
        print(f"My prediction '{predicted_genre}', Actual genre '{test[:, 2]}'")





