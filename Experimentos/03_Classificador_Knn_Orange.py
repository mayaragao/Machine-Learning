####################################################################
# Experimento 03 - CLASSIFICADOR KNN PARA CONJUNTO ORANGE
#
# Modelo preditivo de "churn" para uma empresa de telecomunicações
####################################################################

import pandas            as pd
import matplotlib.pyplot as plt

from sklearn.neighbors     import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

####################################################################
# Experimento 04 - VALIDAÇÃO CRUZADA
#####################################################################

from sklearn.model_selection import train_test_split, cross_val_score

#--------------------------------------------------------------------
# Ler o arquivo CSV com os dados do conjunto IRIS
#--------------------------------------------------------------------

dados = pd.read_csv('Orange_Telecom_Churn_Data.csv')

#--------------------------------------------------------------------
# Explorar os dados
#--------------------------------------------------------------------

print('\n Imprimir o conjunto de dados:\n', dados)

print('\n Imprimir o cnjunto de dados transposto\n para visualizar os nomes de todas as colunas:\n', dados.T)

print('\n Imprimir os tipos de cada variável:\n', dados.dtypes)

print('\n Identificar as variaveis categóricas:\n')

variaveis_categoricas = [
    x for x in dados.columns if dados[x].dtype=='object' or x == 'area_code'
    ]

print(variaveis_categoricas)

print('\n Verificar a cardinalidade de cada variavel categórica:\n')
# obs: Cardinalidade = qtd de valores distindos que a variavel pode assumir

for v in variaveis_categoricas:
    print('\n %15s:' %v, '%4d categorias' %len(dados[v].unique()))
    print(dados[v].unique(), '\n ')

    
#-------------------------------------------------------------------
# Executar pré processamento dos dados:
#-------------------------------------------------------------------    
# Variaveis categóricas:
#    
# state --> não ordinal com 51 categorias  --> DESCARTAR
# area_code --> não ordinal com 3 categorias --> ONE-HOT ENCODING
# phone_numer --> não ordinal com 5000 categorias --> DESCARTAR
# intl_plan --> binaria  --> BINARIZAR (mapear para 0 ou 1)
# voice_mail_plan --> binaria --> BNZARIZAR
#-------------------------------------------------------------------

print('\n Descartar as variaveis de cardinalidade muito alta:\n')

dados = dados.drop(['state', 'phone_number'], axis=1)
print (dados.T)

print('\n Aplicar one-hot encoding nas variaveis com 3 ou mais categorias:\n')

dados = pd.get_dummies(dados,columns=['area_code'])
print(dados.T)

print('\n Aplicar binarização nas variaveis com duas categorias:\n')

binarizador = LabelBinarizer()

for v in ['intl_plan', 'voice_mail_plan']:
    dados[v] = binarizador.fit_transform(dados[v])
    
print(dados.T)
print('\n Para contabilizar quantidade de amostras de cada classe:\n', dados['churned'].value_counts())


# obs: Se houver algum atributo muito diferente no caso de churn para o 
# caso de nao churn, significa que o atributo é relevante para o modelo

print('\n Verificar o valor médio de cada atributo em cada classe:\n')

print(dados.groupby(['churned']).mean().T)

    
#-------------------------------------------------------------------    
# Plotar diagrama de dispersão por classe   
#-------------------------------------------------------------------    

atributo1 = 'total_day_minutes'
atributo2 = 'number_customer_service_calls'

cores = ['red' if x else 'green' for x in dados['churned']]

grafico = dados.plot.scatter(
    atributo1,
    atributo2,
    c= cores,
    s = 10,
    marker ='o',
    alpha = 0.4,
    figsize = (14,8)
    )

plt.show()


#-------------------------------------------------------------------    
# Selecionar os atributos que serão utilizados pelo classificador
#
# colocar comentário para ir selecionando os melhores atributos
#-------------------------------------------------------------------    

atributos_selecionados = [
    #'account_length',
    'intl_plan',
    'voice_mail_plan',
    'number_vmail_messages',
    'total_day_minutes',
    #'total_day_calls',
    'total_day_charge',
    #'total_eve_minutes',
    #'total_eve_calls',
    'total_eve_charge',
    'total_night_minutes',
    #'total_night_calls',
    'total_night_charge',
    #'total_intl_minutes',
    'total_intl_calls',
    'total_intl_charge',
    'number_customer_service_calls',
    #'area_code_408',
    #'area_code_415',
    #'area_code_510'
    'churned'
    ]

dados = dados[atributos_selecionados]

#------------------------------------------------------------------------
# Embaralhar o conjunto de dados para que a divisao entre os dados 
# de treino e de teste esteja isento de qualquer viés de seleção.
#------------------------------------------------------------------------

dados_embaralhados = dados.sample(frac=1, random_state=12344)

#-------------------------------------------------------------------------
# Criar os arrays X e Y separando os atributos e o alvo
#-------------------------------------------------------------------------

x = dados_embaralhados.loc[:,dados_embaralhados.columns != 'churned'].values
y = dados_embaralhados.loc[:,dados_embaralhados.columns == 'churned'].values


#-------------------------------------------------------------------------
# Separa X e Y em conjunto de treino e conjunto de teste
#-------------------------------------------------------------------------


# q = 4000 # qtd de amostras selecionadas para treinamento


# x_treino = x[:q,:]
# y_treino = y[:q].ravel() 

#  método .ravel usado para ajustar as dimensoes que estavam diferentes


# x_teste = x[q:,:]
# y_teste = y[q:].ravel()

#-------------------------------------------------------------------------
# Experimento 04 -> fazendo o split dos dados:
#-------------------------------------------------------------------------

# train test split desempenha mesma funcao que feita manualmente acima
x_treino, x_teste, y_treino, y_teste = train_test_split(
    x,
    y.ravel(),
    train_size=4000, # separando 4000 amostras para treinamento
    random_state=777,
    shuffle=True, # embaralha o conjunto de dados
    # shuffle=False, # nao embarallha os dados, no caso novamente, pois ja estao embaralhados em x e em y.
    # test_size=0.33, # separando 1/3 para ser de teste
    )


#-------------------------------------------------------------------------
# Ajustar a escala dos atributos nos conjuntos de treino e de teste
#-------------------------------------------------------------------------

ajustador_de_escala = MinMaxScaler()
ajustador_de_escala.fit(x_treino)

x_treino = ajustador_de_escala.transform(x_treino)
x_teste = ajustador_de_escala.transform(x_teste)

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
for k in range(1,26,2): # de 2 em 2 unidades
    
    classificador = KNeighborsClassifier(
        n_neighbors = k,
        weights = 'distance', #default é 'uniform' -> esta fazendo pouca diferente
        p = 1 #distancia de manhatan -> default é '2' (euclidiana)
        
        # Métrica p=1 está sendo mais vantajosa
        )
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
    
#-------------------------------------------------------------------------
# Verificar a variação da acurácia com o número de vizinhos 
# usando VALIDAÇÃO CRUZADA
#-------------------------------------------------------------------------


print('\n Variação da acurácia, UTILIZANDO VALIDAÇÃO CRUZADA:')

for k in range(1,26,2):
    # instanciando o classificador:
    classificador = KNeighborsClassifier(
        n_neighbors = k,
        weights = 'distance', 
        p = 1 
        )
    
    scores = cross_val_score(classificador,
                    x,
                    y.ravel(),
                    cv=8,
                    )
    print(
        'k = %2d' % k,
        'scores =', scores,
        'acurácia média = %6.1f' % (100*sum(scores)/8)
        )