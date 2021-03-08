############################################################################################# print( ' Coef. Pearson', col ,'= %6.4f' % pearsonr(dados[col],dados['target'])[0])
# REGRESSOR PARA ESTIMAR O VALOR DE UM IMOVEL
############################################################################################# print( ' Coef. Pearson', col ,'= %6.4f' % pearsonr(dados[col],dados['target'])[0])

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

#from sklearn.preprocessing import Normalizer, RobustScaler

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler, QuantileTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, mean_squared_log_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, HistGradientBoostingRegressor


#--------------------------------------------------------------------
# Ler os dados de treino e teste dos arquivos csv da pasta 'data'
#--------------------------------------------------------------------

dados_treino = pd.read_csv("data/conjunto_de_treinamento.csv")
dados_teste = pd.read_csv("data/conjunto_de_teste.csv")
dados_exemplo_resposta =  pd.read_csv("data/exemplo_arquivo_respostas.csv")


dados = dados_treino
dados_TESTE = dados_teste

# Tratando a coluna diferenciais:

dados['diferenciais'] =  dados['churrasqueira']+  dados['playground']+dados['s_jogos'] + dados['quadra']+dados['sauna']+dados['s_ginastica']+dados['s_festas']
dados_TESTE['diferenciais'] =  dados_TESTE['churrasqueira'] +  dados_TESTE['playground'] +dados_TESTE['s_jogos']+ dados_TESTE['quadra'] +dados_TESTE['sauna']+dados_TESTE['s_ginastica']  +dados_TESTE['s_festas']

# retirando algumas variaveis dos dados:
    
dados = dados_treino.drop(['Id','estacionamento', 'churrasqueira','playground','quadra', 's_festas', 's_jogos','sauna','s_ginastica'], axis=1)
dados_TESTE = dados_teste.drop(['Id','estacionamento', 'churrasqueira','playground','quadra', 's_festas', 's_jogos','sauna', 's_ginastica'], axis=1)

colunas = dados.columns

#-----------------------------------------------------------------------------------
# Analisar os valores do dataframe
#-----------------------------------------------------------------------------------

print('\nColunas do conjunto de treinamento:\n', colunas)

print("\ntipo:\n", dados['tipo'].value_counts())
print("\nbairro:\n", dados['bairro'].value_counts())
print("\nvendedor:\n", dados['tipo_vendedor'].value_counts())


print("\ntipo:\n", dados['tipo'].value_counts())
print("\nbairro:\n", dados['bairro'].value_counts())
print("\nvendedor:\n", dados['tipo_vendedor'].value_counts())


print("\ntipo:\n", dados_TESTE['tipo'].value_counts())

#-------------------------------------------------------------------
# Executar pré processamento dos dados:
#-------------------------------------------------------------------    
# Variaveis categóricas:
#    
# tipo --> não ordinal com 4 categorias --> ONE-HOT ENCODING ?
# bairro --> não ordinal com 66 categorias 
# vendedor --> binaria  --> BINARIZAR (mapear para 0 ou 1)
# diferenciais --> DESCARTAR
#-------------------------------------------------------------------


binarizador = LabelBinarizer()

for t in ['tipo_vendedor']:
    dados[t] = binarizador.fit_transform(dados[t])
    dados_TESTE[t] = binarizador.fit_transform(dados_TESTE[t])


#------------------------------------------------------------------------------  
# Codificando o tipo do imovel com one-hot-encoding
#------------------------------------------------------------------------------

dados = pd.get_dummies(dados, columns=['tipo'])
dados = dados[dados['tipo_Quitinete']==0]
dados = dados.drop(['tipo_Quitinete'], axis=1)

dados = dados[dados['tipo_Loft']==0]
dados = dados.drop(['tipo_Loft'], axis=1)

dados_TESTE = pd.get_dummies(dados_TESTE, columns=['tipo'])
dados_TESTE = dados_TESTE.drop(['tipo_Loft'], axis=1)

#------------------------------------------------------------------------------  
#  Descartar as variaveis de cardinalidade muito alta
#------------------------------------------------------------------------------

print('\n Para contabilizar a cardinalidade das variaveis:\n')
print('Cardinalidade da variavel bairro: \n',len(dados['bairro'].unique().tolist()))
print('Cardinalidade da variavel diferenciais: \n',len(dados['diferenciais'].unique().tolist()))

#------------------------------------------------------------------------------
# Variável bairro, será transformada em algum valor numério usando LabelEncoder
#------------------------------------------------------------------------------

dados_bairro = dados['bairro']
dados_TESTE_bairro = dados_TESTE['bairro']


label_encoder = LabelEncoder()
dados['bairro'] = label_encoder.fit_transform(dados['bairro'])
dados_TESTE['bairro'] = label_encoder.fit_transform(dados_TESTE['bairro'])

# Normalizando dados do LabelEncoder

mean = dados['bairro'].mean()
std = dados['bairro'].std()
dados['bairro'] = (dados['bairro'] - mean) / std


mean = dados_TESTE['bairro'].mean()
std = dados_TESTE['bairro'].std()
dados_TESTE['bairro'] = (dados['bairro'] - mean) / std


dados = dados.drop(['tipo_vendedor'], axis=1)
dados_TESTE = dados_TESTE.drop(['tipo_vendedor'], axis=1)


#------------------------------------------------------------------------------
# Plotando o diagrama e coef. de Pearson para cada atributo dos dados:
#------------------------------------------------------------------------------

colunas = dados.columns

print('\n         Coluna   Coef. Pears.   Confianca')
print('\n --------------   ------------   ---------')

for col in colunas:
    dados.plot.scatter(x = col, y = 'preco', title = '\n Grafico de dispersão para coluna ' + col)
    print( ' %14s   %12.4f   %9.4f' % (col, pearsonr(dados[col],dados['preco'])[0],  pearsonr(dados[col],dados['preco'])[1]))
   

#-----------------------------------------------------------------------------------
# Analisando as relações entre algumas variaveis discretas e o Valor médio de Venda
#-----------------------------------------------------------------------------------

variaveis_discretas = ['quartos', 'vagas', 'piscina', 'vista_mar']

for col in variaveis_discretas:
    plt.figure(figsize = (10,6))
    plt.title('Relação entre '+ col+' vs preco')
    sns.barplot(x = col, y = 'preco', data = dados)
    plt.show()
   
    
#------------------------------------------------------------------------------
# Distribuição das variáveis contínuas:
#------------------------------------------------------------------------------

variaveis_continuas = ['area_util', 'area_extra', 'preco']

for col in variaveis_continuas:
    
    plt.figure(figsize = (8,6))
    plt.title('Distribuição da '+ col)
    sns.distplot(dados[col]);
    

# É possível observar pelo grafico que a há poucos imóveis com
# área extra acima de 1000m, e area_util acima de 750m aproximadamente


plt.figure(figsize = (8,6))
plt.title('Distribuição da area_util')
sns.distplot(dados[dados['area_util']<500]['area_util']);

plt.figure(figsize = (8,6))
plt.title('Distribuição da area_extra')
sns.distplot(dados[dados['area_extra']<100]['area_extra']);


plt.figure(figsize = (8,6))
plt.title('Distribuição do preco')
sns.distplot(dados[dados['preco']<5*pow(10,6)]['preco']);


#------------------------------------------------------------------------
# Analisando oa quantidade de casas e o valor médio das casas por bairro
#------------------------------------------------------------------------

plt.figure(figsize = (8,16))
plt.title('Número de casas por bairro')
sns.countplot(y = dados_bairro)
plt.show()


plt.figure(figsize = (8,16))
plt.title('Valor médio das casas por bairro')
sns.barplot(x = dados[dados['preco']<2*pow(10,6)]['preco'], y = dados_bairro)
plt.show()

#------------------------------------------------------------------------
#Removendo outliers para o preco -> ignorando esses valores muito altos
#------------------------------------------------------------------------

dados_baixos = dados[dados['preco']<=(1.0*pow(10,6))]
dados_baixos = dados_baixos[dados_baixos['area_extra']<200]
dados_baixos = dados_baixos[dados_baixos['area_util']<500]


x = dados_baixos.drop(['preco'], axis=1).to_numpy()
y = dados_baixos['preco'].to_numpy()


x_TESTE = dados_TESTE.iloc[:,:].to_numpy()

y_TESTE = dados_teste.iloc[:,0] #dataframe com coluna index para adicionar a resposta

#--------------------------------------------------------------------
# Definindo métrica de cálculo RMSPE
#--------------------------------------------------------------------

def rmspe(y, y_resposta):
    '''
    Compute Root Mean Square Percentage Error between two arrays.
    '''
    loss = np.sqrt(np.mean(np.square(((y - y_resposta) / y)), axis=0))

    return loss


#--------------------------------------------------------------------
# Separar x e y em conjunto teste e treinamento 
#--------------------------------------------------------------------

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x,
    y,
    test_size=0.20, 
    random_state=0, # sorteio aleatorio
    )


# Ajustar a escala dos atributos 

# Melhores resultados com:
escala = QuantileTransformer()

#2:escala= RobustScaler()
#3: escala = Standart
escala.fit(x_treino)

x_treino = escala.transform(x_treino)
x_teste = escala.transform(x_teste)

x_TESTE = escala.transform(x_TESTE)

#--------------------------------------------------------------------
# Treinar um regressor LINEAR
#--------------------------------------------------------------------

regressor_linear = LinearRegression()
regressor_linear = regressor_linear.fit(x_treino,y_treino)

y_resposta_treino = regressor_linear.predict(x_treino)
y_resposta_teste = regressor_linear.predict(x_teste)

# Calcular as métricas e comparar os resultados

mse_in = mean_squared_error(y_treino, y_resposta_treino)
rmse_in = math.sqrt(mse_in)
r2_in = r2_score(y_treino, y_resposta_treino)
rmspe_in = rmspe(y_treino, y_resposta_treino)

mse_out = mean_squared_error(y_teste, y_resposta_teste)
rmse_out = math.sqrt(mse_out)
r2_out = r2_score(y_teste, y_resposta_teste)
rmspe_out = rmspe(y_teste, y_resposta_teste)


print(' ')
print(' REGRESSOR LINEAR')
print(' ')

print(' Métrica   DENTRO da amostra   FORA da amostra ')
print(' -------   -----------------   --------------- ')

print(' %7s   %17.4f   %15.4f ' % ( 'mse' , mse_in, mse_out) ) 
print(' %7s   %17.4f   %15.4f ' % ( 'rmse' , rmse_in, rmse_out) )
print(' %7s   %17.4f   %15.4f ' % ( 'r2' , r2_in, r2_out) )
print(' %7s   %17.4f   %15.4f ' % ( 'rmspe' , rmspe_in, rmspe_out) )


#-------------------------------------------------------------------------
# Treinar e testear um regressor KNN para varios valores do parametro k
#-------------------------------------------------------------------------

print(' ')
print(' REGRESSOR KNN')
print(' ')

print('  K    DENTRO da amostra   FORA da amostra ')
print(' ---   -----------------   --------------- ')

for k in range(61,76,2):
    regressor_knn = KNeighborsRegressor(
        n_neighbors = k,
        p=1,
        n_jobs=4,
        algorithm = 'kd_tree',
        weights = 'distance', #zera os valores dentro da amostra
        )
    
    regressor_knn = regressor_knn.fit(x_treino,y_treino)

    y_resposta_treino = regressor_knn.predict(x_treino)
    y_resposta_teste = regressor_knn.predict(x_teste)
        
    mse_in = mean_squared_error(y_treino, y_resposta_treino)
    rmse_in = math.sqrt(mse_in)
    r2_in = r2_score(y_treino, y_resposta_treino)
    rmspe_in = rmspe(y_treino, y_resposta_treino)
    
    mse_out = mean_squared_error(y_teste, y_resposta_teste)
    rmse_out = math.sqrt(mse_out)
    r2_out = r2_score(y_teste, y_resposta_teste)
    rmspe_out = rmspe(y_teste, y_resposta_teste)

    print(' %3d   %17.4f   %15.4f ' % ( k , rmspe_in, rmspe_out) ) 


#-------------------------------------------------------------------------
# Treinar e testear um regressor POLINOMIAL para graus de 1 a 5
#-------------------------------------------------------------------------

print(' ')
print(' REGRESSOR POLINOMIAL DE GRAU K')
print(' ')

print('  K    Nº at   DENTRO da amostra   FORA da amostra ')
print(' ---   -----   -----------------   --------------- ')

for k in range(1,4):
    
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
    rmspe_in = rmspe(y_treino, y_resposta_treino)
    
    mse_out = mean_squared_error(y_teste, y_resposta_teste)
    rmse_out = math.sqrt(mse_out)
    r2_out = r2_score(y_teste, y_resposta_teste)
    rmspe_out = rmspe(y_teste, y_resposta_teste)
    
    n_atributos = x_treino_poly.shape[1]    

    print(' %3d   %5d   %17.4f   %15.4f ' % ( k , n_atributos, rmspe_in, rmspe_out) ) 

print('\n Ou seja, para uma regressao de grau 2, o modelo é melhor ajustado, contudo, para grau maior que 2, o modelo está em overfitting, pois tem poucas amostras para aperfeçoar o modelo.') 


#-------------------------------------------------------------------------
# Treinar e testear um regressor RIDGE para graus de 1 a 5
#-------------------------------------------------------------------------

print(' ')
print(' REGRESSOR POLINOMIAL DE GRAU K COM REGULARIZAÇÃO RIDGE (L2):')
print(' ')

print('  K    Nº at   DENTRO da amostra   FORA da amostra ')
print(' ---   -----   -----------------   --------------- ')

#for k in range(3,6):
k=4   
pf = PolynomialFeatures(degree=k)

pf = pf.fit(x_treino)
x_treino_poly = pf.transform(x_treino)
x_teste_poly = pf.transform(x_teste)

#testar qual alpha é possivel obter melhor resultado

regressor_ridge = Ridge(alpha=10.0)
regressor_ridge = regressor_ridge.fit(x_treino_poly,y_treino)

#quanto maior o alpha maior a regularização (underfitting)

y_resposta_treino = regressor_ridge.predict(x_treino_poly)
y_resposta_teste = regressor_ridge.predict(x_teste_poly)

mse_in = mean_squared_error(y_treino, y_resposta_treino)
rmse_in = math.sqrt(mse_in)
r2_in = r2_score(y_treino, y_resposta_treino)
rmspe_in = rmspe(y_treino, y_resposta_treino)

mse_out = mean_squared_error(y_teste, y_resposta_teste)
rmse_out = math.sqrt(mse_out)
r2_out = r2_score(y_teste, y_resposta_teste)
rmspe_out = rmspe(y_teste, y_resposta_teste)

n_atributos = x_treino_poly.shape[1]    

print(' %3d   %5d   %17.4f   %15.4f ' % ( k , n_atributos, rmspe_in, rmspe_out) ) 



#----------------------------------------------------------------------
# Treinar e testar um regressor SGD com função de perda 'squared_loss'
#----------------------------------------------------------------------

print(' ')
print(' REGRESSOR SGD:')
print(' ')

regressor_sgd = SGDRegressor(
    loss='squared_loss',
    alpha=0,
    penalty='elasticnet',
    tol=1e-6,
    l1_ratio= 0.2,
    learning_rate='adaptive',
    max_iter=1000000,
    )
regressor_sgd = regressor_sgd.fit(x_treino,y_treino)

y_resposta_treino = regressor_sgd.predict(x_treino)
y_resposta_teste = regressor_sgd.predict(x_teste)
    

print(' Métrica   DENTRO da amostra   FORA da amostra ')
print(' -------   -----------------   --------------- ')

mse_in = mean_squared_error(y_treino, y_resposta_treino)
rmse_in = math.sqrt(mse_in)
r2_in = r2_score(y_treino, y_resposta_treino)
rmspe_in = rmspe(y_treino, y_resposta_treino)

mse_out = mean_squared_error(y_teste, y_resposta_teste)
rmse_out = math.sqrt(mse_out)
r2_out = r2_score(y_teste, y_resposta_teste)
rmspe_out = rmspe(y_teste, y_resposta_teste)

print(' %7s   %17.4f   %15.4f ' % ( 'mse' , mse_in, mse_out) ) 
print(' %7s   %17.4f   %15.4f ' % ( 'rmse' , rmse_in, rmse_out) )
print(' %7s   %17.4f   %15.4f ' % ( 'r2' , r2_in, r2_out) )
print(' %7s   %17.4f   %15.4f ' % ( 'rmspe' , rmspe_in, rmspe_out) )

#----------------------------------------------------------------------
# Treinar e testar um regressorRandomForest
#----------------------------------------------------------------------


print(' ')
print(' REGRESSOR RANDOM FOREST:')
print(' ')

rf = RandomForestRegressor(
    n_estimators=2000,
    random_state=0,
    min_samples_leaf=2,
    n_jobs=10,
    criterion='mse',
    verbose =0,
    )

rf.fit(x_treino, y_treino)

y_resposta_treino = rf.predict(x_treino)
y_resposta_teste = rf.predict(x_teste)

print(' Métrica   DENTRO da amostra   FORA da amostra ')
print(' -------   -----------------   --------------- ')

mse_in = mean_squared_error(y_treino, y_resposta_treino)
rmse_in = math.sqrt(mse_in)
r2_in = r2_score(y_treino, y_resposta_treino)
medae_in = median_absolute_error(y_treino, y_resposta_treino)
msle_in = mean_squared_log_error(y_treino, y_resposta_treino)
rmspe_in = rmspe(y_treino, y_resposta_treino)

mse_out = mean_squared_error(y_teste, y_resposta_teste)
rmse_out = math.sqrt(mse_out)
r2_out = r2_score(y_teste, y_resposta_teste)
medae_out = median_absolute_error(y_teste, y_resposta_teste)
msle_out = mean_squared_log_error(y_teste, y_resposta_teste)
rmspe_out = rmspe(y_teste, y_resposta_teste)

print(' %7s   %17.4f   %15.4f ' % ( 'mse' , mse_in, mse_out) ) 
print(' %7s   %17.4f   %15.4f ' % ( 'rmse' , rmse_in, rmse_out) )
print(' %7s   %17.4f   %15.4f ' % ( 'r2' , r2_in, r2_out) )
print(' %7s   %17.4f   %15.4f ' % ( 'MedAE:' , round(medae_in,2), round(medae_out,2)))
print(' %7s   %17.4f   %15.4f ' % ( 'MSLE:' , round(msle_in,2), round(msle_out,2)))
print(' %7s   %17.4f   %15.4f ' % ( 'rmspe' , rmspe_in, rmspe_out) )

#----------------------------------------------------------------------
# Treinar e testar um regressor AdaBoost
#----------------------------------------------------------------------


print(' ')
print(' REGRESSOR ADA BOOST:')
print(' ')

ab = AdaBoostRegressor(n_estimators= 100,random_state=0, loss='linear', learning_rate= pow(10, -20))

ab = ab.fit(x_treino, y_treino)

y_resposta_treino = ab.predict(x_treino)
y_resposta_teste = ab.predict(x_teste)


print(' Métrica   DENTRO da amostra   FORA da amostra ')
print(' -------   -----------------   --------------- ')

mse_in = mean_squared_error(y_treino, y_resposta_treino)
rmse_in = math.sqrt(mse_in)
r2_in = r2_score(y_treino, y_resposta_treino)
medae_in = median_absolute_error(y_treino, y_resposta_treino)
msle_in = mean_squared_log_error(y_treino, y_resposta_treino)
rmspe_in = rmspe(y_treino, y_resposta_treino)

mse_out = mean_squared_error(y_teste, y_resposta_teste)
rmse_out = math.sqrt(mse_out)
r2_out = r2_score(y_teste, y_resposta_teste)
medae_out = median_absolute_error(y_teste, y_resposta_teste)
msle_out = mean_squared_log_error(y_teste, y_resposta_teste)
rmspe_out = rmspe(y_teste, y_resposta_teste)

print(' %7s   %17.4f   %15.4f ' % ( 'mse' , mse_in, mse_out) ) 
print(' %7s   %17.4f   %15.4f ' % ( 'rmse' , rmse_in, rmse_out) )
print(' %7s   %17.4f   %15.4f ' % ( 'r2' , r2_in, r2_out) )
print(' %7s   %17.4f   %15.4f ' % ( 'MedAE:' , round(medae_in,2), round(medae_out,2)))
print(' %7s   %17.4f   %15.4f ' % ( 'MSLE:' , round(msle_in,2), round(msle_out,2)))
print(' %7s   %17.4f   %15.4f ' % ( 'rmspe' , rmspe_in, rmspe_out) )


#----------------------------------------------------------------------
# Treinar e testar um regressor HistGradientBoosting
#----------------------------------------------------------------------


print(' ')
print(' REGRESSOR HIST GRADIENT BOOSTING:')
print(' ')

hgb = HistGradientBoostingRegressor(l2_regularization=12.0,max_iter=70,learning_rate=0.1, loss='least_absolute_deviation')

hgb = hgb.fit(x_treino, y_treino)

y_resposta_treino = hgb.predict(x_treino)
y_resposta_teste = hgb.predict(x_teste)


print(' Métrica   DENTRO da amostra   FORA da amostra ')
print(' -------   -----------------   --------------- ')

mse_in = mean_squared_error(y_treino, y_resposta_treino)
rmse_in = math.sqrt(mse_in)
r2_in = r2_score(y_treino, y_resposta_treino)
medae_in = median_absolute_error(y_treino, y_resposta_treino)
msle_in = mean_squared_log_error(y_treino, y_resposta_treino)
rmspe_in = rmspe(y_treino, y_resposta_treino)


mse_out = mean_squared_error(y_teste, y_resposta_teste)
rmse_out = math.sqrt(mse_out)
r2_out = r2_score(y_teste, y_resposta_teste)
medae_out = median_absolute_error(y_teste, y_resposta_teste)
msle_out = mean_squared_log_error(y_teste, y_resposta_teste)
rmspe_out = rmspe(y_teste, y_resposta_teste)

print(' %7s   %17.4f   %15.4f ' % ( 'mse' , mse_in, mse_out) ) 
print(' %7s   %17.4f   %15.4f ' % ( 'rmse' , rmse_in, rmse_out) )
print(' %7s   %17.4f   %15.4f ' % ( 'r2' , r2_in, r2_out) )
print(' %7s   %17.4f   %15.4f ' % ( 'MedAE:' , round(medae_in,2), round(medae_out,2)))
print(' %7s   %17.4f   %15.4f ' % ( 'MSLE:' , round(msle_in,2), round(msle_out,2)))
print(' %7s   %17.4f   %15.4f ' % ( 'rmspe' , rmspe_in, rmspe_out) )


#----------------------------------------------------------------------
# Aplicar o modelo com RMSPE = 23.4483 no conjunto de teste:
#----------------------------------------------------------------------


#y_resposta_TESTE = regressor_sgd.predict(x_TESTE)

y_resposta_TESTE = hgb.predict(x_TESTE)
d = {'Id': y_TESTE , 'preco': y_resposta_TESTE}
df = pd.DataFrame(data=d)
#df = df.T


df.to_csv('out.csv', index=False)

dados_output =  pd.read_csv("out.csv")
