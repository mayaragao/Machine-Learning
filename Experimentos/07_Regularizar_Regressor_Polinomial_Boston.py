############################################################################################# print( ' Coef. Pearson', col ,'= %6.4f' % pearsonr(dados[col],dados['target'])[0])
# REGRESSOR LINEAR vs KNN vs POLINOMIAL - CONJUNTO BOSTON
############################################################################################# print( ' Coef. Pearson', col ,'= %6.4f' % pearsonr(dados[col],dados['target'])[0])

import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso

#--------------------------------------------------------------------
#Ler as amostras da planilha excel e gravas como dataframe Pandas
#--------------------------------------------------------------------

dados = pd.read_excel("D02_Boston.xlsx")

#--------------------------------------------------------------------
# Transferir valores dos atributos e rótulos para x e y
#--------------------------------------------------------------------

x = dados.iloc[:,1:-1].to_numpy()
y = dados.iloc[:,-1].to_numpy()

#--------------------------------------------------------------------
# Separar x e y em conjunto teste e treinamento
#--------------------------------------------------------------------

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x,
    y,
    test_size=200, 
    random_state=0, # sorteio aleatorio
    # test_size=0.33, # separando 1/3 para ser de teste
    )

#--------------------------------------------------------------------
# Ajustar a escala dos atributos 
#--------------------------------------------------------------------

escala = StandardScaler()

escala.fit(x_treino)

x_treino = escala.transform(x_treino)
x_teste = escala.transform(x_teste)

#--------------------------------------------------------------------
# Treinar um regressor LINEAR
#--------------------------------------------------------------------

regressor_linear = LinearRegression()
regressor_linear = regressor_linear.fit(x_treino,y_treino)

#--------------------------------------------------------------------
# Obter as respostas do regressor linear dentro e fora da amostra
#--------------------------------------------------------------------

y_resposta_treino = regressor_linear.predict(x_treino)
y_resposta_teste = regressor_linear.predict(x_teste)

#--------------------------------------------------------------------
# Calcular as métricas e comparar os resultados
#--------------------------------------------------------------------

mse_in = mean_squared_error(y_treino, y_resposta_treino)
rmse_in = math.sqrt(mse_in)
r2_in = r2_score(y_treino, y_resposta_treino)

mse_out = mean_squared_error(y_teste, y_resposta_teste)
rmse_out = math.sqrt(mse_out)
r2_out = r2_score(y_teste, y_resposta_teste)

print(' ')
print(' REGRESSOR LINEAR')
print(' ')

print(' Métrica   DENTRO da amostra   FORA da amostra ')
print(' -------   -----------------   --------------- ')

print(' %7s   %17.4f   %15.4f ' % ( 'mse' , mse_in, mse_out) ) 
print(' %7s   %17.4f   %15.4f ' % ( 'rmse' , rmse_in, rmse_out) )
print(' %7s   %17.4f   %15.4f ' % ( 'r2' , r2_in, r2_out) )


#-----------------------------------------------------------------------------
# Plotar diagrama de dispersao entre a resposta correta X resposta do modelo
#-----------------------------------------------------------------------------

# plt.scatter(x=y_teste, y = y_resposta_teste)


#-------------------------------------------------------------------------
# Treinar e testear um regressor KNN para varios valores do parametro k
#-------------------------------------------------------------------------

print(' ')
print(' REGRESSOR KNN')
print(' ')

print('  K    DENTRO da amostra   FORA da amostra ')
print(' ---   -----------------   --------------- ')

for k in range(1,21):
    regressor_knn   = KNeighborsRegressor(
        n_neighbors = k,
        #weights     = 'uniform',
        weights     = 'distance', #zera os valores dentro da amostra
        )
    
    regressor_knn = regressor_knn.fit(x_treino,y_treino)

    y_resposta_treino = regressor_knn.predict(x_treino)
    y_resposta_teste = regressor_knn.predict(x_teste)
        
    mse_in = mean_squared_error(y_treino, y_resposta_treino)
    rmse_in = math.sqrt(mse_in)
    r2_in = r2_score(y_treino, y_resposta_treino)
    
    mse_out = mean_squared_error(y_teste, y_resposta_teste)
    rmse_out = math.sqrt(mse_out)
    r2_out = r2_score(y_teste, y_resposta_teste)


    print(' %3d   %17.4f   %15.4f ' % ( k , rmse_in, rmse_out) ) 


#-------------------------------------------------------------------------
# Treinar e testear um regressor POLINOMIAL para graus de 1 a 5
#-------------------------------------------------------------------------

print(' ')
print(' REGRESSOR POLINOMIAL DE GRAU K')
print(' ')

print('  K    Nº at   DENTRO da amostra   FORA da amostra ')
print(' ---   -----   -----------------   --------------- ')

for k in range(1,6):
    
    pf = PolynomialFeatures(degree=k)
    
    pf = pf.fit(x_treino)
    x_treino_poly = pf.transform(x_treino)
    x_teste_poly = pf.transform(x_teste)
    
    regressor_linear = LinearRegression()
    regressor_linear = regressor_linear.fit(x_treino_poly,y_treino)

    y_resposta_treino = regressor_linear.predict(x_treino_poly)
    y_resposta_teste = regressor_linear.predict(x_teste_poly)
    
    mse_in = mean_squared_error(y_treino, y_resposta_treino)
    rmse_in = math.sqrt(mse_in)
    r2_in = r2_score(y_treino, y_resposta_treino)
    
    mse_out = mean_squared_error(y_teste, y_resposta_teste)
    rmse_out = math.sqrt(mse_out)
    r2_out = r2_score(y_teste, y_resposta_teste)

    n_atributos = x_treino_poly.shape[1]    

    print(' %3d   %5d   %17.4f   %15.4f ' % ( k , n_atributos, rmse_in, rmse_out) ) 

print('\n Ou seja, para uma regressao de grau 2, o modelo é melhor ajustado, contudo, para grau maior que 2, o modelo está em overfitting, pois tem poucas amostras para aperfeçoar o modelo.') 


#-------------------------------------------------------------------------
# Treinar e testear um regressor RIDGE para graus de 1 a 5
#-------------------------------------------------------------------------

print(' ')
print(' REGRESSOR POLINOMIAL DE GRAU K COM REGULARIZAÇÃO RIDGE (L2):')
print(' ')

print('  K    Nº at   DENTRO da amostra   FORA da amostra ')
print(' ---   -----   -----------------   --------------- ')

for k in range(1,4):
    
    pf = PolynomialFeatures(degree=k)
    
    pf = pf.fit(x_treino)
    x_treino_poly = pf.transform(x_treino)
    x_teste_poly = pf.transform(x_teste)
    
    #testar qual alpha é possivel obter melhor resultado
    
    regressor_ridge = Ridge(alpha=50.0)
    regressor_ridge = regressor_ridge.fit(x_treino_poly,y_treino)

    #quanto maior o alpha maior a regularização (underfitting)

    y_resposta_treino = regressor_ridge.predict(x_treino_poly)
    y_resposta_teste = regressor_ridge.predict(x_teste_poly)
    
    mse_in = mean_squared_error(y_treino, y_resposta_treino)
    rmse_in = math.sqrt(mse_in)
    r2_in = r2_score(y_treino, y_resposta_treino)
    
    mse_out = mean_squared_error(y_teste, y_resposta_teste)
    rmse_out = math.sqrt(mse_out)
    r2_out = r2_score(y_teste, y_resposta_teste)

    n_atributos = x_treino_poly.shape[1]    

    print(' %3d   %5d   %17.4f   %15.4f ' % ( k , n_atributos, rmse_in, rmse_out) ) 

print(' Melhora dos resultados a partir de k=2.')


#-------------------------------------------------------------------------
# Treinar e testear um regressor LASSO para graus de 1 a 5
#-------------------------------------------------------------------------


print(' ')
print(' REGRESSOR POLINOMIAL DE GRAU K COM REGULARIZAÇÃO LASSO (L1):')
print(' ')

print('  K    Nº at   DENTRO da amostra   FORA da amostra ')
print(' ---   -----   -----------------   --------------- ')

for k in range(1,4):
    
    pf = PolynomialFeatures(degree=k)
    
    pf = pf.fit(x_treino)
    x_treino_poly = pf.transform(x_treino)
    x_teste_poly = pf.transform(x_teste)
    
    #testar qual alpha é possivel obter melhor resultado
    
    regressor_lasso = Lasso(alpha=0.1, max_iter = 100000)
    regressor_lasso = regressor_lasso.fit(x_treino_poly,y_treino)

    #quanto maior o alpha maior a regularização (underfitting)

    y_resposta_treino = regressor_lasso.predict(x_treino_poly)
    y_resposta_teste = regressor_lasso.predict(x_teste_poly)
    
    mse_in = mean_squared_error(y_treino, y_resposta_treino)
    rmse_in = math.sqrt(mse_in)
    r2_in = r2_score(y_treino, y_resposta_treino)
    
    mse_out = mean_squared_error(y_teste, y_resposta_teste)
    rmse_out = math.sqrt(mse_out)
    r2_out = r2_score(y_teste, y_resposta_teste)

    n_atributos = x_treino_poly.shape[1]    

    print(' %3d   %5d   %17.4f   %15.4f ' % ( k , n_atributos, rmse_in, rmse_out) ) 

print('\n A regularizacao lasso demora mais para convergir. e ajusta melhor para funcoes de grau maior.\n')


#-------------------------------------------------------------------------
# Comparar os coeficientes da cada regularizacao
#-------------------------------------------------------------------------

print('\nCoeficientes RIDGE:\n', regressor_ridge.coef_)
print('\nCoeficientes LASSO:\n', regressor_lasso.coef_)

# Nesse caso, a maior parte dos coeficientes converge pra 0(coeficientes nulos)
# o que pode melhorar o tempo de processamento em comparação ao regressor L2.

# Contudo, em resultados absolutos, normalmente a regularizacao L2, 
# consegue ser melhor que a regularização L1.