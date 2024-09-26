#editar conforme for necessário retirei daqui https://gist.github.com/sebleier/554280
stopWords=["i", "me", "my", "we", "our", "ours", "ourselves", "you", 
           "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", 
           "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", 
           "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
           "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
           "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a",
           "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", 
           "of", "at", "by", "for", "with", "about", "against", "between", "into",
           "through", "during", "before", "after", "above", "below", "to", "from", "up",
           "down", "in", "out", "on", "off", "over", "under", "again", "further", "then",
           "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
           "each", "few", "more", "most", "other", "some", "such", "nor", "not",
           "only", "own", "same", "so", "than", "too", "very", "can",
           "will", "just", "don", "should", "now"]

class PreProcessing:
    def __init__(self,filename):
        self.dados_filmes = []
        with open("./data/"+filename, 'r', encoding='utf-8') as file:
            # Lendo o arquivo linha por linha
            for linha in file:
                # Remover espaços em branco no início e fim (incluindo quebras de linha)
                linha = linha.strip()
                # Separar os dados usando o tab ('\t') como delimitador
                filme = linha.split('\t')
                # Adicionar a lista de filmes ao array 2D
                self.dados_filmes.append(filme)
        
        #print(self.dados_filmes[0],"\n")
    
    def cleaningText(self):
        for i in range(len(self.dados_filmes)):
            # Separar o plot (que está na 5ª coluna) em palavras
            texto = self.dados_filmes[i][4].split(" ")
            # Filtrar palavras que não estão nas stopwords
            newText = [word for word in texto if word.lower() not in stopWords]
            # Juntar as palavras filtradas de volta em uma string
            self.dados_filmes[i][4] = " ".join(newText)
        
        #print(self.dados_filmes[0])
    
    def returnCleanText(self):
        self.cleaningText()
        return self.dados_filmes