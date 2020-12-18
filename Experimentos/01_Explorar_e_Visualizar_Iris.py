####################################################################
# Experimento 01 - Explorar e visualizar o conjunto de dados IRIS
####################################################################

import pandas as pd
import matplotlib.pyplot as plt

#Carregar o conjunto de dados IRIS do csv

dados = pd.read_csv('Iris_Data.csv', delimiter=',', decimal='.')
# dados = pd.read_csv('Iris_Data.csv')

print('\n Exibindo informações sbre o conjunto de dados')
print('\n Primeiros dados da tabela: \n')
print(dados.head(n=6))

print('\n\n Ultimos dados da tabela: \n')
print(dados.tail(n=4))

print('\n\n Dimensões do conjunto de dados: \n')
print(dados.shape)
print('\n O conjunto tem', dados.shape[0], 'amostras com', dados.shape[1], 'variaveis.')

print('\n\n Tipos das variaveis: \n')
print(dados.dtypes)

print('\n\n Para criar um dataframe somente com dados das petalas: \n Exibindo as 5 primeiras linhas da tabela: \n')
dados_petalas = dados[ ['petal_length', 'petal_width'] ]
print(dados_petalas.head())

print('\n\n Para mudar o nome da espécie: ')
print('\n Retirando o prefixo Iris- das espécies: \n')

# dados['species'] = dados['species'].str.replace('Iris-', '')

# maneira mais flexivel de modificar os valores do dataframe:
dados['species'] = dados['species'].apply(lambda r: r.replace('Iris-', ''))

print(dados['species'].head())

print('\n\n Contabilizar a quantidade de amostras de cada variavel da tabela: \n')
print(dados['species'].value_counts())

print('\n\n Exibir informações estatísticas sobre os dados: \n')
print(dados.describe())

# informações pontuais sobre os dados:

print('\n\n Exibir média dos dados: \n')
print(dados.mean())


# mediana oferece um valor típico pouco sensível a outliers!
print('\n\n Exibir mediana das colunas: \n')
print(dados.median())

print('\n\n Exibir desvio padrão das colunas: \n')
print(dados.std())

print('\n\n Exibir média dos dados agrupado por espécie: \n')
print(dados.groupby('species').mean())

print('\n\n Montar tabela de estatísticas personalizadas: \n')
tabela = dados.groupby('species').agg(
    {
     'petal_length': ['median', 'mean', 'std'],
     'petal_width': ['mean', 'std']
     }
    )
print(tabela)

print('\n\n Montar tabela com informações estatísticas de todos atributos: \n')
tabela = dados.groupby('species').agg(
    {
     x: ['median', 'mean', 'std'] for x in dados.columns if x != 'species'
     }
    )

#convertendo o resultado para string
print(tabela.to_string())


#---------------------------------------------
# Exibir graficos
#---------------------------------------------


print('\n\n Visualizar o histograma de uma variavel: \n')

grafico = dados['petal_length'].plot.hist(bins=40)

grafico.set(
    title = 'DISTRIBUIÇÃO DO COMPRIMENTO DA PÉTALA',
    xlabel = 'Comprimento da Pétala (cm)',
    ylabel = 'Número de amostras'
    )

plt.show()


print('\n\n Visualizar o diagrama de dispersão entre duas variáveis: \n')

grafico = dados.plot.scatter('petal_width', 'petal_length')

grafico.set(
    title = 'DISPERSÃO LARGURA vs COMPRIMENTO DA PÉTALA',
    xlabel = 'Largura da Pétala (cm)',
    ylabel = 'Comprimento da Pétala (cm)'
    )

plt.show()



#---------------------------------------------------------
# Separar os atributos e o alvo em dataframes distintos
#---------------------------------------------------------

# Formato genérico:
# atributos = dados.iloc[lin1:lin2,col1:col2]

# Nesse caso vai de 10 a 19, pois o primeiro valor é INCLUSIVO, e o ultimo é EXCLUSIVO

# atributos = dados.iloc[10:20,0:4]

atributos = dados.iloc[:,0:4]
rotulos = dados.iloc[:,4]

# -1 : os atributos sao todas as colunas exceto a ultima
# -1 : os rótulos esão na ultima coluna

atributos = dados.iloc[:,0:-1]
rotulos = dados.iloc[:,-1]

#---------------------------------------------------------
# Montar lista com os valores distindos dos rótulos (classes)
#---------------------------------------------------------

classes = dados['species'].unique().tolist()

#---------------------------------------------------------
# Montar mapa de cores associando cada classe a uma cor
#---------------------------------------------------------

mapa_de_cores = ['purple', 'green', 'blue']
cores_das_amostras = [mapa_de_cores[classes.index(r)] for r in rotulos]

#---------------------------------------------------------
# Visualizar a matriz de dispersão dos atributos
#---------------------------------------------------------

# Para todos atributos:
    
pd.plotting.scatter_matrix(
    atributos,
    c = cores_das_amostras,
    figsize = (11,11),
    marker='v',
    s =30, #tamanho do marcador
    alpha = 0.5, #bom para amostras que se sobrepoem
    diagonal = 'hist', #pode ser 'hist' ou 'kde'
    hist_kwds={ 'bins':20 }    
    )

plt.suptitle('MATRIZ DE DISPERSÃO DOS ATRIBUTOS', y=0.9, fontsize='xx-large')
plt.show()


# Para alguns atributos:

pd.plotting.scatter_matrix(
    atributos[['petal_width', 'petal_length']],
    c = cores_das_amostras,
    figsize = (11,11),
    marker='v',
    s = 25, #tamanho do marcador
    alpha = 0.5, #bom para amostras que se sobrepoem
    diagonal = 'hist', #pode ser 'hist' ou 'kde'
    hist_kwds={ 'bins':20 }    
    )

plt.show()


#-------------------------------------------------------------------
# Visualizar um gráfico de dispersão 3D entre 3 atributos
#-------------------------------------------------------------------

# escolher as variaveis de cada eixo

eixo_x = 'sepal_length'
eixo_y = 'petal_length'
eixo_z = 'petal_width'

# criar uma figura

figura = plt.figure(figsize = (15,12))

# criar um gráfico 3D dentro da figura => default é 2D, 1(n da linha) 1(n coluna) 1(n de ordem do grafico)

grafico = figura.add_subplot(111,projection='3d')

# Exemplo: Um grafico do lado do outro

# grafico1 = figura.add_subplot(121,projection='3d')
#grafico2 = figura.add_subplot(122,projection='3d')

grafico.scatter(
    dados[eixo_x],
    dados[eixo_y],
    dados[eixo_z],
    c = cores_das_amostras,
    marker='o',
    s = 40, 
    alpha = 0.5, 
    )


plt.suptitle('GRÁFICO DE DISPERSÃO 3D', y=0.9, fontsize='xx-large')
plt.show()

