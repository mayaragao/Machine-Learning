##############################################################################
# Experimento 05 - EXPLORANDO E VISUALIZANDO O CONJUNTO "BOSTON" (REGRESSÃO)
#############################################################################

import pandas            as pd
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

#--------------------------------------------------------------------
# Importar conjunto de dados de planilha excel para dataframe Pandas
#--------------------------------------------------------------------

dados = pd.read_excel("D02_Boston.xlsx")

#--------------------------------------------------------------------
# Descartar primeira coluna (unamed pois era o indice no excel)
#--------------------------------------------------------------------

dados = dados.iloc[:,1:]

#--------------------------------------------------------------------
# Verificar as colunas disponiveis
#--------------------------------------------------------------------

colunas = dados.columns

print("Colunas disponíveis:")
print(colunas)


#--------------------------------------------------------------------
# Plotar diagrama de dispersao entre cada atributo e o alvo
#
# Para um atributo:
#--------------------------------------------------------------------
    
atributo_selecionado = 'CRIM'
alvo = 'target'

# Diagrama de dispersão
dados.plot.scatter(x=atributo_selecionado, y= alvo)

# Coeficiente de Pearson: funcao perasonr retorna uma tupla, o 
# primeiro elemento é o coeficiente desejado, e o pi value, reflete 
# a confiança nos dados da estatística.

coeficiente_pearson = pearsonr(dados[atributo_selecionado],dados['target'])
print( 'Coef. Pearson = %.4f' % coeficiente_pearson[0])

#-----------------------------------------------------------------------
# Plotando o diagrama e coef. de Pearson para cada atributo dos dados:
#-----------------------------------------------------------------------

for col in colunas:
    dados.plot.scatter(x=col, y= alvo, title='\n Grafico de dispersão para coluna ' + col)
    
    print( ' Coef. Pearson %10s = %6.4f' % (col, pearsonr(dados[col],dados['target'])[0]) )
    # print(' Confiança     ', col ,'= %.4f' % pearsonr(dados[col],dados['target'])[1])

#-----------------------------------------------------------------------
# Explorar correlaçoes mútuas entre os atributos: 
# 
# OBS: Correlação nao significa causalidade!
#-----------------------------------------------------------------------

# Escolhendo nossos atributos para relacionar

atributo_1 = 'LSTAT'
atributo_2 = 'RM'
dados.plot.scatter(x=atributo_1, y= atributo_2, title= atributo_1 +' vs ' + atributo_2)
print( ' Correlaçao ',atributo_1+" x "+atributo_2 ,' %6.4f' % pearsonr(dados[atributo_1],dados[atributo_2])[0])
   
# ou seja, apesar dessas colunas serem as que mais tem relacao com o target,
# ambas tambem tem correlação forte, entao talvez, nao seja interessante adicionar
# ambas variaveis para o modelo preditivo. 

# Para selecionar as melhores variavies, nao necessariamente as mais correlacionadas
# com o alvo sao as melhores a serem selecionadas, principalmente, se elas já tem
# correlacao entre sí, melhor escolher variaveis que tem pouca correlação.