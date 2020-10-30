def sum_(A, B):
    """Retorna soma dos vetores A,B se são de mesma "ordem", mesmo
    número de elementos se vetores unidimensionais e mesmo número
    de linhas e mesmo número de colunas para cada linha, se vetores
    bidimensionais (matrizes)."""

    C = [] 
    number_rows_A = len(A) #salva número de linhas de A
    number_rows_B = len(B) #salva número de linhas de B
    
    if number_rows_A == number_rows_B: #se mesmo número de linhas
        if isinstance(A[0], float) or isinstance(A[0], int): #se vetor unidimensional
            return [A[i] + B[i] for i in range(number_rows_A)]
        else: #senão, vetor bidimensional
            C = [] #matriz resultado da soma
            for i in range(number_rows_A): #para cada linha
                number_columns_Ai = len(A[i]) #salva número de colunas de A[i]
                number_columns_Bi = len(B[i]) #salva número de colunas de B[i]
                if number_columns_Ai == number_columns_Bi: #se mesmo número de colunas
                    row = [A[i][j] + B[i][j] for j in range(number_columns_Ai)]
                    C.append(row) #salva na matriz
                else:
                    print("\nSum Error. (number_columns_A[{}]: {}) != (number_columns_B[{}]: {})."
                            .format(i, number_columns_Ai, i, number_columns_Bi))
                    exit(1)
            return C
    else:
        print("\nSum Error. (number_rows_A: {}) != (number_rows_B: {})."
                .format(number_rows_A, number_rows_B))
        exit(1)    

def subtraction(A, B):
    """Retorna subtração dos vetores A,B se são de mesma "ordem",
    mesmo número de elementos se vetores unidimensionais e mesmo
    número de linhas e mesmo número de colunas para cada linha,
    se vetores bidimensionais (matrizes)."""

    C = [] 
    number_rows_A = len(A) #salva número de linhas de A
    number_rows_B = len(B) #salva número de linhas de B
    
    if number_rows_A == number_rows_B: #se mesmo número de linhas
        if isinstance(A[0], float) or isinstance(A[0], int): #se vetor unidimensional
            return [A[i] - B[i] for i in range(number_rows_A)]
        else: #senão, vetor bidimensional
            C = [] #matriz resultado da soma
            for i in range(number_rows_A): #para cada linha
                number_columns_Ai = len(A[i]) #salva número de colunas de A[i]
                number_columns_Bi = len(B[i]) #salva número de colunas de B[i]
                if number_columns_Ai == number_columns_Bi: #se mesmo número de colunas
                    row = [A[i][j] - B[i][j] for j in range(number_columns_Ai)]
                    C.append(row) #salva na matriz
                else:
                    print("\nSubtraction Error. (number_columns_A[{}]: {}) != (number_columns_B[{}]: {})."
                            .format(i, number_columns_Ai, i, number_columns_Bi))
                    exit(1)
            return C
    else:
        print("\nSubtraction Error. (number_rows_A: {}) != (number_rows_B: {})."
                .format(number_rows_A, number_rows_B))
        exit(1)

def multiplication(A, B):
    """Se Am,n * Bn,p, retorna matriz multiplicação da ordem m,p.
    Se A é escalar e B matriz, retorna matriz B multiplicada por
    escalar, se A é escalar e B vetor unidimensional, retorna vetor
    B multiplicado por escalar."""
    
    if isinstance(A, float) or isinstance(A, int):
        if isinstance(B[0], float) or isinstance(B[0], int):
            return [A * B[i] for i in range(len(B))]
        return [[A * B[i][j] for j in range(len(B[i]))] for i in range(len(B))]

    number_columns_A = len(A[0])
    number_rows_B = len(B)

    if number_columns_A == number_rows_B:
        return [[sum(i * j for i, j in zip(row, column)) for column in zip(*B)]
                for row in A]
    else:
        print("\nMultiplication Error. (number_columns_A: {}) != (number_rows_B: {})."
                .format(number_columns_A, number_rows_B))
    exit(1)

def transpose(A):
    """Se A for lista vazia retorna lista vazia,
    se A for vetor unidimensional ou bidimensional (matriz), 
    retorna a transposta."""

    AT = [] #inicia transposta como lista vazia

    if len(A) == 0: #se número de linhas igual a zero
        return AT #retorna lista vazia
    else:
        if isinstance(A[0], float) or isinstance(A[0], int): #se vetor unidimensional
            AT = [[elm] for elm in A] #salva cada elemento como linha
        else:
            for i in range(len(A)): #para cada linha
                for j in range(len(A[i])): #para cada coluna da linha
                    if len(AT) < j + 1: #se coluna ainda não existe
                        AT.append([A[i][j]]) #salva como nova linha
                    else:
                        AT[j].append(A[i][j]) #salva coluna

    return AT #retorna tranposta
