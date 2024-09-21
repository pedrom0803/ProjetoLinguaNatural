from preProcessing import PreProcessing
from nn import SimpleFFNN

class Train:
    def __init__(self, data_to_train : str,nnFile : str,layer_size, learning_rate : float , epochs : int) -> None:
        self.file=data_to_train
        self.model= SimpleFFNN(layer_size)
        self.export_nn=nnFile
        self.learning_rate=learning_rate
        self.epochs = epochs
        
    def train(self):
        x = self.file[:, [0, 1, 3, 4]]  # Colunas 0, 1, 3 e 4
        y = self.file[:, 2]  # Coluna 2
        self.model.train(x, y, self.epochs, self.learning_rate)    
        self.model.save_model(self.export_nn)