import os
from nn import SimpleFFNN
from train import Train
from preProcessing import PreProcessing
import numpy as np
from testModel import TestModel

def split_array(data: np.ndarray, train_size: float = 0.8):
    """
    Divide um array 2D em dois arrays: um com train_size dos dados e outro com o restante.
    
    :param data: O array 2D a ser dividido.
    :param train_size: A proporção de dados a serem usados para o primeiro array (default é 0.8).
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
    nnFile="ficheiro_com_novo_modelo"
    model=None

    nodes_outuput=[len(genres)]
    
    #limpesa de ficheiro de treino
    data="data/train"
    
    print("\033[34mPre Processing the data\n\033[0m")
    
    pp=PreProcessing(data+".txt")
    clean_data=pp.returnCleanText()
    
    print("\033[32mPre Processing Completed!\n\033[0m")
    

    # Dividir os dados em treino e teste
    data_to_train, data_to_test = split_array(clean_data)

    # Carregar o modelo se ele já existir
    if os.path.isfile(nnFile+".pkl"):
        print("\033[34mLoading Model\n\033[0m")
        
        model=SimpleFFNN.load_model(nnFile+".pkl")
        
        print("\033[32mLoading Completed!\n\033[0m")
    # Se não existir, cria e treina um novo modelo   
    else:
        print("\033[34mCreating a new Model\n\033[0m")
        
        newNNFile="data/ficheiro_com_novo_modelo"
        layer_hidden = [5] # adicionar o numero de nos por camada,hidden, que bem se entender([5]->5 nos na camada hidden1; [3,6]-> 3 na camada hidden 1 e 6 na hidden 2;...)
        learning_rate = 0.01
        epochs=1
        model = Train(data_to_train,newNNFile,layer_hidden,nodes_outuput, learning_rate, epochs)
        model.train()
        
        print("\033[32mModel Created!\n\033[0m")
        
    model_teste= TestModel("ficheiro_com_novo_modelo")
    
    # Testar e comparar labels
    #model_teste.test_with_label(data_to_test)
    
    # Testar e apenas escrever resultados no ficheiro 'results.txt'
    # pp_no_label = PreProcessing("test_no_labels")
    # clean_data_no_label = pp_no_label.returnCleanText()
    # model_teste.test_without_labels(clean_data_no_label)




