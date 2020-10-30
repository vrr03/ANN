import networks.MLSN as nwk
import matplotlib.pyplot as plt

def read_data_file(name):
    f = open(name, 'r') #abre arquivo de dados
    dataset = [] #conjunto de dados do arquivo de dados
    classes = [] #lista de classes do conjunto de dados
    for line in f: #para cada linha no arquivo
        l = line.split(',') #salva linha do arquivo como lista, separando por ','
        elm = [float(x) for x in l[:-1]] #salva entradas na lista de elementos da linha
        s = l[-1][:-1] #copia string de classe de saída sem '\n'
        elm.append(s) #salva string na lista de elementos
        if s not in classes: #se classe ainda não salva na lista de classes
            classes.append(s) #salva classe
        dataset.append(elm) #salva lista de elementos no conjunto de dados
    f.close() #fecha arquivo
    return (dataset, classes) #retorna tupla de lista de conjunto de dados e classes

(dataset, classes) = read_data_file("iris.data")

k = 10 #número de folds
eta = 0.01 #passo de aprendizagem
alfa = 0.99 #termo momentum
epochs = 100 #número de épocas de treinamento

sizes = [len(dataset[0]) - 1, 8, len(classes)] #modelo da rede
ann = nwk.MLSN(sizes) #inicia rede
# C = ann.stratified_cross_validation(classes, dataset) #retorna a união das listas de custos por época de todos os folds
C = ann.stratified_cross_validation(classes, dataset, k, eta, alfa, epochs) #retorna a união das listas de custos por época de todos os folds

plt.plot([x for x in range(len(C))], C)
plt.title('Iris')
plt.grid(True)
plt.ylabel('União dos Custos')
plt.xlabel('União das Épocas')
plt.show()