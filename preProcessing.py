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
import re
from datetime import datetime

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
    
    def _month_to_number(self,month_name):
        try:
            return datetime.strptime(month_name, "%b").month  # Abbreviated month names (e.g., "Feb")
        except ValueError:
            try:
                return datetime.strptime(month_name, "%B").month  # Full month names (e.g., "February")
            except ValueError:
                return None  # Return None for invalid month names

    def _handle_two_digit_year(self,year):
        # Convert two-digit years to four-digit format (assuming 1900s or 2000s)
        year = int(year)
        return year + 2000 if year < 100 else year

    def standardize_dates(self,text):
        # Handle cases like "February 8th, 2013" or "Feb 8th"
        text = re.sub(r'(\b[A-Za-z]+) (\d{1,2})(?:st|nd|rd|th)?(?:,)? (\d{4})?', 
                    lambda match: f" {int(match.group(2)):02d}/{self._month_to_number(match.group(1)):02d}/"
                                    f"{match.group(3)} " if self._month_to_number(match.group(1)) is not None and match.group(3) else 
                                    f" {int(match.group(2)):02d}/{self._month_to_number(match.group(1)):02d} " 
                                    if self._month_to_number(match.group(1)) is not None else match.group(0), 
                    text)
        
        # Handle cases like "8th-Feb" or "8-Feb-2013"
        text = re.sub(r'(\d{1,2})(?:st|nd|rd|th)?[-/.](\b[A-Za-z]+)\b[-/.]?(\d{2,4})?', 
                    lambda match: f" {int(match.group(1)):02d}/{self._month_to_number(match.group(2)):02d}/"
                                    f"{self._handle_two_digit_year(match.group(3))} " if self._month_to_number(match.group(2)) is not None and match.group(3) else 
                                    f" {int(match.group(1)):02d}/{self._month_to_number(match.group(2)):02d} " 
                                    if self._month_to_number(match.group(2)) is not None else match.group(0), 
                    text)
        
        # Handle formats like "02/08/13" (assuming this is MM/DD/YY)
        text = re.sub(r'(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})', 
                    lambda match: f" {int(match.group(2)):02d}/{int(match.group(1)):02d}/{self._handle_two_digit_year(match.group(3))} ", 
                    text)
        
        # Handle formats like "September 1876" -> "09/1876"
        text = re.sub(r'(\b[A-Za-z]+) (\d{4})', 
                    lambda match: f" {self._month_to_number(match.group(1)):02d}/{match.group(2)} " if self._month_to_number(match.group(1)) is not None else match.group(0), 
                    text)
        
        return text
    
    def _cleaningText(self,plot_index):
        for i in range(len(self.dados_filmes)):
            # Separar o plot (que está na 5ª coluna) em palavras
            text_data_standard=self.standardize_dates(self.dados_filmes[i][plot_index])
            texto = text_data_standard.split(" ")
            # Filtrar palavras que não estão nas stopwords
            newText = [word for word in texto if word.lower() not in stopWords]
            # Juntar as palavras filtradas de volta em uma string
            self.dados_filmes[i][plot_index] = " ".join(newText)
        
        #print(self.dados_filmes[0])
    
    def returnCleanText(self,plot_index=4):
        self._cleaningText(int(plot_index))
        return self.dados_filmes
    
    def returnCleanInputText(texto):
        text = texto.split(" ")
        newText = [word for word in text if word.lower() not in stopWords]
        return " ".join(newText)  # Junta as palavras da lista de volta em uma string
    
        
        
# pp=PreProcessing("train.txt")
# print(pp.standardize_dates("Ola ze September 2010 adeus"))