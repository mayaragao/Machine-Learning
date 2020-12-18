####################################################################
# Experimento 02 - CLASSIFICADOR KNN PARA O CONJUNTO IRIS
####################################################################

# Bibliotecas:

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Leitura do arquivo CSV com os dados do treinamento

dados = pd.read_csv('Iris_Data.csv')

#------------------------------------------------------------------------
# Randomizar as amostras para que a divisao entre os dados de treino e
# de teste esteja isento de qualquer viés de seleção.
#------------------------------------------------------------------------

dados_embaralhados = dados.sample(frac=1, random_state=12345)

#-------------------------------------------------------------------------
# Criar os arrays X e Y para os conuntos de teste e de treinamento
# Utilizando 100 amostras para o treinamento e 50 para testes
# ou seja, 2/3 para treinamento tipico para conjutos de dados pequenos
#-------------------------------------------------------------------------

x_treino = dados_embaralhados.iloc[:100,:-1].values
y_treino = dados_embaralhados.iloc[:100, -1].values


x_teste = dados_embaralhados.iloc[100:,:-1].values
y_teste = dados_embaralhados.iloc[100:, -1].values

#-------------------------------------------------------------------------
# Criar um classificador KNN -> Importar do pacote Neighbors do sklearn
#-------------------------------------------------------------------------

classificador = KNeighborsClassifier(n_neighbors=10)
classificador = classificador.fit(x_treino, y_treino)


#-------------------------------------------------------------------------
# Obter as respostas do classificaor no mesmo conjunto onde foi treinado
#-------------------------------------------------------------------------

y_resposta_treino = classificador.predict(x_treino)


#-------------------------------------------------------------------------
# Obter as respostas do classificaor no conjunto teste
#-------------------------------------------------------------------------

y_resposta_teste = classificador.predict(x_teste)

#-------------------------------------------------------------------------
# Verificando a acurácia do classificador:
#-------------------------------------------------------------------------

print('\n DESEMPENHO DENTRO DA AMOSTRA DE TREINO')

total = len(y_treino)
acertos = sum(y_resposta_treino==y_treino)
erros = sum(y_resposta_treino!=y_treino)

print('\n Total de amostras:', total)
print(' Respostas corretas:', acertos)
print(' Respostas erradas:', erros)

acuracia = acertos/total

print(' Acurácia = %.1f %%' %(100*acuracia))

print('\n DESEMPENHO NA AMOSTRA DE TESTE')

total = len(y_teste)
acertos = sum(y_resposta_teste==y_teste)
erros = sum(y_resposta_teste!=y_teste)

print('\n Total de amostras:', total)
print(' Respostas corretas:', acertos)
print(' Respostas erradas:', erros)

acuracia = acertos/total

print(' Acurácia = %.1f %%' %(100*acuracia))

#-------------------------------------------------------------------------
# Verificar a variação da acurácia com o número de vizinhos
#-------------------------------------------------------------------------

print('\n Variação da acurácia, sendo K o nº de vizinhos')
print('\n  K  TREINO  TESTE ')
print(' --  ------  -----')
for k in range(1,15):
    
    classificador = KNeighborsClassifier(n_neighbors=k)
    classificador = classificador.fit(x_treino, y_treino)

    
    y_resposta_treino = classificador.predict(x_treino)
    y_resposta_teste = classificador.predict(x_teste)
    
    acuracia_treino = sum(y_resposta_treino==y_treino)/len(y_treino)
    acuracia_teste = sum(y_resposta_teste==y_teste)/len(y_teste)
    
    print(
        "%3d"%k,
        "%6.1f" %(100*acuracia_treino),
        "%6.1f" %(100*acuracia_teste)
        )