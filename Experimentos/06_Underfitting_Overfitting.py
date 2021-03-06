#################################################################################
# EXPERIMENTO 6.2 - UNDERFITTING E OVERFITTING
#################################################################################

import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Modificar o grau polinomial da regressao para analisar o melhor grau para ajuste
grau_polinomial = 3 # grau da regressao polinomial

# Aumentar o numero de amostras pode ajudar no overfitting.
numero_amostras = 2000 

#-------------------------------------------------------------------------
# Definir uma função senoidal base para gerar as amostras
#-------------------------------------------------------------------------

#Criando um array com pontos equidistantes
x_base = np.linspace(0.00, 1.00, num=101).reshape(-1,1)
print("dimensões do vetor x_base", x_base.shape)

#funcao senoidal no valor de 0 a 1 -> periodo igual a 2PI
y_base = np.sin(2 * np.pi * x_base)
print("dimensões do vetor y_base", x_base.shape)

#--------------------------------------------------------------------------------
# Gerar amostras aleatorias com desvio padrao de 0.2 em torno da função senoidal
#--------------------------------------------------------------------------------

np.random.seed(0)

x = np.random.rand(numero_amostras, 1) 
#rand -> sorteia valores de forma homogenea entre 0 e 1

y = np.sin(2 * np.pi * x) + 0.20*np.random.randn (numero_amostras,1)
#randn -> gera numeros seguindo uma distribuição normal entre 0 e 1

# print(' x= \n', x, '\n y= \n', y)

#--------------------------------------------------------------------------------
# Dividir as amostras entre conjunto de treinamento e conjunto de teste
#--------------------------------------------------------------------------------

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x,
    y,
    test_size=0.5,  #metade das amostras para treinamento
    #random_state=0, # sorteio aleatorio
    )


#--------------------------------------------------------------------------------
# Visualizar as amostras em um grafico de dispersao
#--------------------------------------------------------------------------------

plt.figure(figsize=(9,9))

plt.title("AMOSTRAS DISPONÍVEIS")

plt.plot(
    x_base,
    y_base,
    color = 'gray',
    linestyle = 'dotted',
    label = 'Função alvo (desconhecida)'
    )

plt.scatter(
    x_treino,
    y_treino,
    color = 'green',
    marker= 'o',
    s = 30,
    alpha = 0.5,
    label = 'Amostras de treinamento'
    )

plt.scatter(
    x_teste,
    y_teste,
    color = 'red',
    marker= 'o',
    s = 30,
    alpha = 0.5,
    label = 'Amostras de teste'
    )

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#--------------------------------------------------------------------------------
# Treinar e testar o modelo de regressao polinomial
#--------------------------------------------------------------------------------

# OVERFITTING -> Poucas amostras em relação à complexidade do modelo 

# Instanciar um obj polinomialFeatures

pf = PolynomialFeatures(degree = grau_polinomial)
pf = pf.fit(x_treino)

# Transformar a matriz x incluindo os atributos polinomiais
x_treino_transf = pf.transform(x_treino)
x_teste_transf = pf.transform(x_teste)
x_base_transf = pf.transform(x_base)

print(x_treino_transf)

# Instanciar e treinar o modelo de regressao linear

modelo = LinearRegression()
modelo = modelo.fit(x_treino_transf, y_treino)

# Obter respostas do modelo DENTRO e FORA da amostra

y_resposta_treino = modelo.predict(x_treino_transf)
y_resposta_teste = modelo.predict(x_teste_transf)

y_resposta_base = modelo.predict(x_base_transf) # para plotar a funcao do modelo encontrada

# Calcular métricas de erro das respostas

rmse_in = math.sqrt(mean_squared_error(y_resposta_treino, y_treino))
rmse_out = math.sqrt(mean_squared_error(y_resposta_teste, y_teste))

print("\n rmse_in", rmse_in)
print("\n rmse_out", rmse_out)


#--------------------------------------------------------------------------------
# Visualizar graficamente os resultados
#--------------------------------------------------------------------------------


plt.figure(figsize=(16,9))

quadro1 = plt.subplot(121) #grafico da esquerda
plt.ylim(-1.5, 1.5)

quadro2 = plt.subplot(122) #grafico da direita
plt.ylim(-1.5, 1.5)

# Exibir resultados DENTRO da amostra

quadro1.title.set_text(
    ( "Regressão de grau %d\n" % grau_polinomial ) +
    "Desempenho DENTRO da amostra\n" +
    ( "RMSE: %.4f" % rmse_in )
    )

quadro1.plot(
    x_base,
    y_base,
    color = 'gray',
    linestyle = 'dotted',
    label = 'Função alvo (desconhecida)'
    )

quadro1.scatter(
    x_treino,
    y_treino,
    color = 'green',
    marker= 'o',
    s = 30,
    alpha = 0.5,
    label = 'Amostras de treinamento'
    )

quadro1.scatter(
    x_treino,
    y_resposta_treino,
    color = 'blue',
    marker= 'x',
    s = 30,
    alpha = 0.5,
    label = 'Respostas do modelo'
    )

quadro1.plot(
    x_base,
    y_resposta_base,
    color = 'purple',
    linestyle = 'dotted',
    label = 'Função de decisão'
    )


# Exibir resultados FORA da amostra

quadro2.title.set_text(
    ( "Regressão de grau %d\n" % grau_polinomial ) +
    "Desempenho FORA da amostra\n" +
    ( "RMSE: %.4f" % rmse_out )
    )

quadro2.plot(
    x_base,
    y_base,
    color = 'gray',
    linestyle = 'dotted',
    label = 'Função alvo (desconhecida)'
    )

quadro2.scatter(
    x_teste,
    y_teste,
    color = 'red',
    marker= 'o',
    s = 30,
    alpha = 0.5,
    label = 'Amostras de teste'
    )


quadro2.scatter(
    x_teste,
    y_resposta_teste,
    color = 'blue',
    marker= 'x',
    s = 30,
    alpha = 0.5,
    label = 'Respostas do modelo'
    )

quadro2.plot(
    x_base,
    y_resposta_base,
    color = 'purple',
    linestyle = 'dotted',
    label = 'Função de decisão'
    )

plt.legend()
plt.show()

# o erro mínimo do modelo FORA da amostra ideal, foi no grau 3, pois foi
# o menor rmse obtido. Para grau polinomial 4 ocorre overfitting -> modelo
# muito ajustado aos dados de treinamento