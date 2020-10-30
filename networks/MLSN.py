#Imports:
from math import exp
import algebra.vectorOperations as vOp
import random

random.seed()
# random.seed(7) #gera sempre a mesma sequência

def sigmoid(z):
    """Função de Ativação Sigmóide."""
    return 1./(1. + exp(-z))

# def sigmoid_prime(z):
#     """Função para retornar as derivadas da função Sigmóide"""
#     a = sigmoid(z)
#     return a * (1 - a)

#Classe "Multi-Layer of Sigmoide Neurons" (Multicamadas de Neurônios Sigmóides):
class MLSN:
    def __init__(self, sizes):
        """A lista "sizes" contém o número de neurônios nas respectivas
        camadas da rede. Os vieses e pesos para a rede são inicializados
        aleatoriamente entre -0.3 e 0.3. Note que a primeira camada é
        assumida como uma camada de entrada, e por convenção nós não
        definimos nenhum "bias" para esses neurônios, pois os "biases"
        são usados na computação das saídas das camadas posteriores."""

        self.num_layers = len(sizes) #salva número de camadas
        self.sizes = sizes #salva vetor de tamanhos
        #inicia biases aleatórios:
        # self.biases = [[random.uniform(-1., 1.) for b in range(i)] for i in sizes[1:]]
        self.biases = [[random.uniform(-0.3, 0.3) for b in range(i)] for i in sizes[1:]]
        #inicia weights aleatórios:
        # self.weights = [[[random.uniform(-1., 1.) for w in range(i)] for k in range(j)]
        #                     for j, i in zip(sizes[1:], sizes[:-1])]
        self.weights = [[[random.uniform(-0.3, 0.3) for w in range(i)] for k in range(j)]
                            for j, i in zip(sizes[1:], sizes[:-1])]
    
    def feedforward_compute(self, x):
        """Retorna a saída da rede se 'x' for input de teste."""
        
        a = vOp.transpose(x) #camada de entradas transposta
        
        for b, w in zip(self.biases, self.weights):
            #calcula as somas ponderadas da camada:
            zl = vOp.sum_(vOp.multiplication(w, a), vOp.transpose(b))
            #lista auxiliar de níveis de ativações da camada (Sig(zl))
            al = [sigmoid(zl[k][0]) for k in range(len(zl))]
            a = vOp.transpose(al) #atualiza entradas para a próxima camada, até ser saída

        return vOp.transpose(a) #retorna saída da rede
    
    def evaluate(self, fold, classes, dataset):
        """Avalia conjunto de exemplos referentes ao fold,
        e retorna o número de acertos e número de exemplos."""

        hs = 0 #número de acertos (hits)
        ns = 0 #número de exemplos de teste (samples)
        for p in fold: #para cada índice de padrão no fold
            x = dataset[p][:-1] #copia o vetor de entrada do padrão
            y = [] #inicia vetor de saída do padrão
            for i, c in zip(range(self.sizes[-1]), classes): #para cada (índice de classe, classe)
                if dataset[p][-1] == c: #se encontrou a classe
                    for j in range(self.sizes[-1]): #para cada neurônio de saída
                        if j == i: y.append(1.) #se é o neurônio da classe, salva com 1
                        else: y.append(0.) #senão, salva com 0
                    break #sai do loop
            A = self.feedforward_compute(x) #computa saída da rede para o padrão de entrada
            h_a = 0. #maior ativação
            i_a = -1. #índice de maior ativação 
            for j in range(len(A[0])): #para cada neurônio de saída
                if A[0][j] > h_a: #se nível de ativação do neurônio atual for maior 
                    h_a = A[0][j] #atualiza nível de ativação
                    i_a = j #atualiza índice
            if i_a != -1: #se não houve nenhum erro
                if y[i_a] == 1.: #acertou
                    hs += 1 #incrementa número de acertos
            ns += 1 #incrementa número de exemplos
        
        return (hs, ns)

    def derive_activations(self, l, j, y, a, dS_z, dC_a, dC_a_mem):
        """Retorna as derivadas do custo em relação às ativações dos neurônios
        para o exemplo de treinamento."""

        if (dC_a_mem[l][j] == 1): #se derivada já calculada
            return dC_a[l][j] #retorna valor

        if l == (self.num_layers - 2): #se for a última camada
            dC_a[l][j] = (-2. / self.sizes[-1]) * (y[j] - a[l][j]) #calcula derivada
        else:
            for k in range(self.sizes[l+2]): #para todos os neurônios da próxima camada
                #Calcula derivada:
                dC_a[l][j] += self.weights[l+1][k][j] * dS_z[l+1][k] * self.derive_activations(l+1, k, y, a, dS_z, dC_a, dC_a_mem)
        
        dC_a_mem[l][j] = 1 #marca como calculado

        return dC_a[l][j] #retorna valor

    def backpropagation(self, x, y, a, dS_z):
        """Retorna tupla "(nabla_b, nabla_w)" representando o gradiente para a função de custo Cp.
        "nabla_b" e "nabla_w" são listas de camadas de matrizes semelhantes a "biases" e "weights"."""

        #inicia vetor derivadas do custo em relação aos biases (gradiente dos biases) com 0:
        nabla_b = [[0. for b in range(i)] for i in self.sizes[1:]]
        #inicia vetor derivadas do custo em relação aos pesos (gradiente dos pesos) com 0:
        nabla_w = [[[0. for w in range(i)] for k in range(j)] 
                    for j, i in zip(self.sizes[1:], self.sizes[:-1])]

        #inicia vetor derivadas do custo em relação às ativações:
        dC_a = [[0. for a in range(i)] for i in self.sizes[1:]]
        #inicia memorização do vetor derivadas do custo em relação às ativações:
        dC_a_mem = [[0. for a in range(i)] for i in self.sizes[1:]]

        #Para cada tupla índice de camada e quantidade de neurônios (de trás para frente):
            #l -> [num_layers - 2, ..., 0]
            #nl -> [sizes[num_layers - 1], ..., sizes[1]]
        for l, nl in zip(range(self.num_layers - 2, -1, -1), self.sizes[-1::-1]):
            for j in range(nl): #para cada neurônio da camada
                #Calcula gradiente dos biases da camada:
                nabla_b[l][j] += dS_z[l][j] * self.derive_activations(l, j, y, a, dS_z, dC_a, dC_a_mem)

        #Para cada tupla índice de camada e quantidades de neurônios (de trás para frente) de camada atual e anterior (nj, ni)
            #l -> [num_layers - 2, ..., 0]
            #nj -> [sizes[num_layers - 1], ..., sizes[1]]
            #ni -> [sizes[num_layers - 2], ..., sizes[0]]
        for l, nj, ni in zip(range(self.num_layers - 2, -1, -1), self.sizes[-1::-1], self.sizes[-2::-1]):                                                                     
            for j in range(nj):
                for i in range(ni):
                    if l == 0:
                        #calcula gradiente dos pesos da camada
                        nabla_w[l][j][i] += x[i] * dS_z[l][j] * self.derive_activations(l, j, y, a, dS_z, dC_a, dC_a_mem)
                    else:
                        #calcula gradiente dos pesos da camada
                        nabla_w[l][j][i] += a[l-1][i] * dS_z[l][j] * self.derive_activations(l, j, y, a, dS_z, dC_a, dC_a_mem)
        
        return (nabla_b, nabla_w) #retorna tupla gradiente
    
    def MSE(self, a, y):
        """Retorna custo médio do exemplo de treinamento,
        pelo método de erro quadrático médio."""

        return sum(((yi - ai)**2) for yi, ai in zip(y, a)) / self.sizes[-1]
    
    def feedforward(self, x):
        """Retorna a tupla de níveis de ativação e derivadas
        da função de ativação para todas as somas ponderadas
        dos neurônios da rede, dado um padrão de entrada 'x'."""

        A = [] #lista de níveis de ativação [sigmóide(zl)]
        a = vOp.transpose(x) #camada de entradas transposta
        dS_z = [] #lista de derivadas da função sigmóide

        for b, w in zip(self.biases, self.weights):
            #calcula as somas ponderadas da camada:
            zl = vOp.sum_(vOp.multiplication(w, a), vOp.transpose(b))
            al = [] #lista auxiliar de níveis de ativações da camada (Sig(zl))
            dS_zl = [] #lista auxiliar de derivadas de Sig(zl)
            
            for k in range(len(zl)): #para cada soma ponderada
                al.append(sigmoid(zl[k][0])) #salva nível de ativação do neurônio
                dS_zl.append(al[-1] * (1 - al[-1])) #calcula e salva as derivadas Sig'(zl)
                                                        #na lista auxiliar

            A.append(al) #salva lista de níveis de ativação da camada
            dS_z.append(dS_zl) #salva lista de derivadas Sig'(zl)
            a = vOp.transpose(al) #atualiza entrada para próxima camada, até ser saída

        return (A, dS_z) #retorna tupla de níveis de ativação e derivadas da função de sigmoide
    
    def training(self, fold, classes, dataset, epochs, eta, alfa):
        """Modo batch de treinamento com termo momentum."""

        C = [] #lista de custos médios por epóoca de treinamento
        #inicia variação dos vieses:
        delta_b = [[0. for b in range(i)] for i in self.sizes[1:]]
        #inicia variação do pesos:
        delta_w = [[[0. for w in range(i)] for k in range(j)] 
                    for j, i in zip(self.sizes[1:], self.sizes[:-1])]
        
        for e in range(epochs): #para cada época de treinamento
            Cp = [] #lista auxiliar de custos para cada padão de treinamento

            #inicia vetor derivadas do custo em relação aos biases (gradiente dos biases) com 0:
            nabla_B = [[0. for b in range(i)] for i in self.sizes[1:]]
            #inicia vetor derivadas do custo em relação aos pesos (gradiente dos pesos) com 0:
            nabla_W = [[[0. for w in range(i)] for k in range(j)] 
                    for j, i in zip(self.sizes[1:], self.sizes[:-1])]

            for p in range(len(fold)): #para cada índice de padrão do fold
                x = dataset[fold[p]][:-1] #copia o vetor de entrada do padrão
                y = [] #inicia vetor de saída do padrão
                for i, c in zip(range(self.sizes[-1]), classes): #para cada (índice de classe, classe)
                    if dataset[fold[p]][-1] == c: #se encontrou a classe
                        for j in range(self.sizes[-1]): #para cada neurônio de saída
                            if j == i: y.append(1.) #se é o neurônio da classe, salva com 1
                            else: y.append(0.) #senão, salva com 0
                        break #sai do loop
                (A, dS_z) = self.feedforward(x) #retorna níveis de ativação e derivadas da função
                                                        #de ativação para as somas ponderadas zl
                Cp.append(self.MSE(y, A[-1])) #retorna custo do padrão
                (nabla_b, nabla_w) = self.backpropagation(x, y, A, dS_z) #retorna tupla de vetor gradiente

                #Atualiza vetores gradiente de vieses e pesos:
                for l in range(self.num_layers - 1):
                    nabla_B[l] = vOp.sum_(nabla_B[l], nabla_b[l])
                    nabla_W[l] = vOp.sum_(nabla_W[l], nabla_w[l])

            #Atualiza variação dos vieses e pesos (com termo momentum) e os mesmos:
            for l in range(self.num_layers - 1):
                delta_b[l] = vOp.sum_(vOp.multiplication(eta, nabla_B[l]), vOp.multiplication(alfa, delta_b[l]))
                delta_w[l] = vOp.sum_(vOp.multiplication(eta, nabla_W[l]), vOp.multiplication(alfa, delta_w[l]))
                self.biases[l] = vOp.subtraction(self.biases[l], delta_b[l])
                self.weights[l] = vOp.subtraction(self.weights[l], delta_w[l])
                
            #Atualiza vieses e pesos (sem termo momentum):
            # for l in range(self.num_layers - 1):
            #     self.biases[l] = vOp.subtraction(self.biases[l], vOp.multiplication(eta, nabla_B[l]))
            #     self.weights[l] = vOp.subtraction(self.weights[l], vOp.multiplication(eta, nabla_W[l]))

            C.append(sum(Cp)/len(fold)) #salva a média dos custos médios quadráticos dos exemplos do fold por época
            if C[-1] <= 0.01: #se custo médio final <= 0.01
                break #completa treinamento do fold

        return C #retorna lista de custos
    
    def partition(self, c, k, nck):
        """Retorna lista de listas de índices, mutuamente exclusivas e proporcionais em relação às classes."""

        folds = [] #lista de partições mutuamente exclusivas e proporcionais em relação às classes
        for i in range(k): #para cada partição
            fold = [] #inicia partição de índices
            for j, cj in zip(range(self.sizes[-1]), c): #para cada (índice classe, linha do objeto de classes)
                if cj[1] != 0: #se ainda possui elemento de índice de exemplo
                    random.shuffle(cj[2]) #embaralha lista
                    if cj[1] >= nck: #se possui índices suficiente
                        for z in range(nck): #percorre nck índices
                            fold.append(cj[2][0]) #salva primeiro índice no fold
                            del c[j][2][0] #deleta índice do objeto
                            c[j][1] -= 1 #decrementa um índice
                    else: #senão
                        for z in range(cj[1]): #percorre o resto da lista
                            fold.append(cj[2][0]) #salva primeiro índice no fold
                            del c[j][2][0] #deleta índice do objeto
                            c[j][1] -= 1 #decrementa um índice
            random.shuffle(fold) #embaralha lista
            folds.append(fold) #salva lista
        return folds
    
    def stratified_cross_validation(self, classes, dataset, k=10, eta=0.01, alfa=0.99, epochs=100):
        N = len(dataset) #número de exemplos

        c = [[y, 0, []] for y in classes] #[classe, número de exemplos para a classe, [índices dos exemplos]]

        for i, l in zip(range(N), dataset): #para cada (índice linha do dataset, linha do dataset)
            for j, y in zip(range(self.sizes[-1]), classes): #para cada (índice classe, classe)
                if l[-1] == y: #quando achar a classe
                    c[j][1] += 1 #incrementa número de exemplos
                    c[j][2].append(i) #salva índice do exemplo
                    break
        
        n = int(N / k) #número de índices exemplos por partição
        nck = int(n / self.sizes[-1]) #número de índices de exemplos de cada classe para cada partição

        folds = self.partition(c, k, nck)
        
        #Treinamento:
        C = [] #lista de custos de todas as épocas de treinamento
        for e in range(epochs): #para cada época de treinamento
            Cf = [] #lista de custos finais para os folds  
            for i in range(k-2): #faz treinamento para k-2 folds
                Ci = self.training(folds[i], classes, dataset, epochs, eta, alfa) #lista de custos médios para o treinamento do fold
                Cf.append(Ci[-1]) #salva custo final do fold
                C += Ci #concatena listas
            if sum(Cf)/(k-2) <= 0.01: #se a média dos custos finais dos folds for <= 0.01
                break #termina treinamento
        
        #Teste:
        hs = 0 #número de acertos (hits)
        ns = 0 #número de exemplos de teste (samples)
        for i in range(k-2, k): #testa para os exemplos dos dois últimos folds
            (hsf, nsf) = self.evaluate(folds[i], classes, dataset) #avalia conjunto
            hs += hsf #incrementa número de acertos
            ns += nsf #incrementa número de exemplos
        print("\nAccuracy: ({}/{})*100 = {}%\n".format(hs, ns, (hs/ns)*100)) #imprime acurácia
        
        return C #retorna lista de custos de todas as épocas