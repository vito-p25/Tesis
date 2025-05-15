# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 15:16:29 2025

@author: Victoria
"""

#PROBANDO EL CODIGO DE FUNCIONES 

import codigo_funciones as c_fn 
import numpy as np
import matplotlib.pyplot as plt
import time 
import pandas as pd


inicio = time.time()

#%% DEFINICION DE VARIABLES 

#path= 'C:/Users/Victoria/OneDrive/Documents/Proyecto ANII/CODIGO VERSION 14 DE ABRIL'

year_train = 2017
year_test = 2018
kc_min = 0 
kc_max = 1.4
fh_temporal = np.arange(1,6)
quantiles = np.arange(0.1,1,0.1)
lag = 5
alpha = 0.2


estaciones = ['CAB','CAR','CEN','MIL','PAL','PAY','TAB','TOR']
EST = estaciones[4]

print(f'TRABAJANDO con estacion: {EST}')

#%% SELECCION DE CONJUNTO DE FEATURES PARA FORWARD SELECTION

path = 'C:\\Users\\Victoria\\OneDrive\\Documents\\Proyecto ANII\\CODIGO VERSION 14 DE ABRIL\\dataset'

dataset_dummy = c_fn.dataset(path, fh_temporal[0], EST, year_train, year_test, lag)
features_disponibles = list(dataset_dummy.data_train.columns)

variables_a_testear = [var for var in features_disponibles if var.startswith(('S', 'B', 'C', 'PCA'))] #FILTRACION DE LO QUE QUIERO AGREGAR 


resultados_metricas = {}

for variable in variables_a_testear:
    print(f'\nProbando conjunto: {variable}')
    metricas_fh = []

    for fh in fh_temporal:
        Dataset = c_fn.dataset(path, fh, EST, year_train, year_test, lag)
        X_train, X_test, y_train, y_test = Dataset.get_data([variable])  #solo una variable por vez

        modelo = c_fn.Regresion_cuantil(kc_min, kc_max, quantiles)
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        
        gcs = Dataset.target(Dataset.data_test['gcs'])

        met = c_fn.metricas(pred, y_test, alpha, gcs)
        metricas_fh.append(met.cal_metricas(modelo.quantiles))

    resultados_metricas[variable] = metricas_fh
    

fin = time.time()

print(f'Tiempo de ejecucion {fin - inicio }')


#%% GUARDADO EN CSV LOS DATOS DEL FORWARD SELECTION

filas_metricas = []
filas_confiabilidad = []

for variable, metricas_fh in resultados_metricas.items():
    for i, fh in enumerate(fh_temporal):
        frec_obs, picp, ace, pinaw, crps = metricas_fh[i]

        # Métricas generales por FH
        filas_metricas.append({
            'Variable': variable,
            'FH': fh,
            'PICP': picp,
            'ACE': ace,
            'PINAW': pinaw,
            'CRPS': crps
        })

        # Diagrama de confiabilidad (una fila por cuantil)
        for q, freq in zip(quantiles, frec_obs):
            filas_confiabilidad.append({
                'Variable': variable,
                'FH': fh,
                'Cuantil': q,
                'Freq_obs': freq
            })

df_metricas = pd.DataFrame(filas_metricas)
df_confiabilidad = pd.DataFrame(filas_confiabilidad)

df_metricas.to_csv(f'{path}/metricas_{EST}.csv', index=False)
df_confiabilidad.to_csv(f'{path}/confiabilidad_{EST}.csv', index=False)



#%% GRAFICOS FORWARD SELECTION 

df_metricas1 = pd.read_csv(f'{path}/metricas_{EST}.csv')

conj_f_optima = c_fn.mejores_variables_por_fh(df_metricas1)

c_fn.plot_fam_metrica(df_metricas1, 'CRPS', 'S')
c_fn.plot_fam_metrica(df_metricas1, 'CRPS', 'C')
c_fn.plot_fam_metrica(df_metricas1, 'CRPS', 'B')
c_fn.plot_fam_metrica(df_metricas1, 'CRPS', 'PCA')


c_fn.plot_fam_metrica(df_metricas1, 'PICP', 'S')
c_fn.plot_fam_metrica(df_metricas1, 'PICP', 'C')
c_fn.plot_fam_metrica(df_metricas1, 'PICP', 'B')
c_fn.plot_fam_metrica(df_metricas1, 'PICP', 'PCA')

c_fn.plot_fam_metrica(df_metricas1, 'PINAW', 'S')
c_fn.plot_fam_metrica(df_metricas1, 'PINAW', 'C')
c_fn.plot_fam_metrica(df_metricas1, 'PINAW', 'B')
c_fn.plot_fam_metrica(df_metricas1, 'PINAW', 'PCA')


c_fn.plot_fam_metrica(df_metricas1, 'ACE', 'S')
c_fn.plot_fam_metrica(df_metricas1, 'ACE', 'C')
c_fn.plot_fam_metrica(df_metricas1, 'ACE', 'B')
c_fn.plot_fam_metrica(df_metricas1, 'ACE', 'PCA')



#%%  EXPERIMENTO (TRABAJO ANTECEDENTE 1: ALONSO ) vs CARACTERISTICA OPTIMA POR HORIZONTE TEMPORAL 
conj_features_exp_antedecente = [[],['var'],['var','S6']]
#conj_features_exp_antedecente = [['var','S6']]


fh_temporal = np.arange(1,6)
f_opt_fh = c_fn.feature_opt_fh(df_metricas1,fh_temporal)

m_c1, m_c2,m_c3 , m_f_opt = [],[],[],[]


for fh in fh_temporal:  
    
    Dataset = c_fn.dataset(path, fh, EST, year_train, year_test, lag)
    gcs = Dataset.target(Dataset.data_test['gcs'])

    for i, conj in enumerate(conj_features_exp_antedecente):
        X_train, X_test, y_train, y_test = Dataset.get_data(conj)

        modelo = c_fn.Regresion_cuantil(kc_min, kc_max, quantiles)
        modelo.fit(X_train, y_train)
        X_pred = modelo.predict(X_test)

        met = c_fn.metricas(X_pred, y_test, alpha, gcs)
        met_resultado = met.cal_metricas(modelo.quantiles)

        if i == 0:
            m_c1.append(met_resultado)
        elif i == 1:
            m_c2.append(met_resultado)
        elif i == 2:
            m_c3.append(met_resultado)

    X_train_opt, X_test_opt, y_train_opt, y_test_opt = Dataset.get_data(['var',f_opt_fh[fh - 1]])

    modelo1 = c_fn.Regresion_cuantil(kc_min, kc_max, quantiles)
    modelo1.fit(X_train_opt, y_train_opt)
    X_pred_opt = modelo1.predict(X_test_opt)

    met_opt = c_fn.metricas(X_pred_opt, y_test_opt, alpha, gcs)
    met_resultado_opt = met_opt.cal_metricas(modelo1.quantiles)

    m_f_opt.append(met_resultado_opt)


#%% GRAFICO DE LOS MODELOS CON DISTINTOS CONJUNTOS DE FEATURES


path = 'C:\\Users\\Victoria\\OneDrive\\Documents\\Proyecto ANII\\CODIGO VERSION 14 DE ABRIL'

plt.figure(figsize=(10,6))

plt.plot(fh_temporal,list(zip(*m_c1))[4],label = '[kc(t),...,kc(t-lag)]')
plt.plot(fh_temporal,list(zip(*m_c2))[4],label = f'{conj_features_exp_antedecente[1]}')
plt.plot(fh_temporal,list(zip(*m_c3))[4],label = f'{conj_features_exp_antedecente[2]}')
plt.plot(fh_temporal,list(zip(*m_f_opt))[4],label ='Features optimas por fh')
plt.xlabel('Horizonte temporal (fh)')
plt.xlabel('Horizonte temporal (FH)', fontsize=12)
plt.ylabel('CRPS', fontsize=12)
#plt.legend(title='Variable', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f'{path}/graficos/CRPS_{EST}_distintos_conjuntos.png')



plt.figure(figsize=(10,6))

plt.plot(fh_temporal, list(zip(*m_c1))[3], label='[kc(t),...,kc(t-lag)]')
plt.plot(fh_temporal, list(zip(*m_c2))[3], label=f'{conj_features_exp_antedecente[1]}')
plt.plot(fh_temporal, list(zip(*m_c3))[3], label=f'{conj_features_exp_antedecente[2]}')
plt.plot(fh_temporal, list(zip(*m_f_opt))[3], label='Features óptimas por fh')

plt.xlabel('Horizonte temporal (FH)', fontsize=12)
plt.ylabel('PINAW', fontsize=12)

#plt.legend(title='Variable', bbox_to_anchor=(1.05, 1), loc='upper left')  # Solo esta

plt.tight_layout()
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig(f'{path}/graficos/PINAW_{EST}_distintos_conjuntos.png')



#DIAGRAMA DE CONFIABILIDAD
c_fn.plot_diag_conf(EST,quantiles,fh_temporal,m_c1,'[kc(t),...,kc(t-5)]')
c_fn.plot_diag_conf(EST,quantiles,fh_temporal,m_c2,'[kc(t),...,kc(t-5),var]')
c_fn.plot_diag_conf(EST,quantiles,fh_temporal,m_c3,'[kc(t),...,kc(t-5),var,S6]')
c_fn.plot_diag_conf(EST,quantiles,fh_temporal,m_c1,'Característica optima por fh')






# # #METRICAS: conf,PICP,ACE,PINAW,CRPS

# # # PICP, PINAW y CRPS por variable
# picp = list(zip(*metricas_fh))[1]
# pinaw = list(zip(*metricas_fh))[3]
# crps = list(zip(*metricas_fh))[4]

# # color = '#1f77b4'
        
# plt.figure(figsize=(8, 6))
# plt.plot(np.arange(1,6), crps, '*-', color=color, linewidth=1, markersize=8)
# plt.xlabel('Horizonte temporal (min)', fontsize=12)
# plt.ylabel('crps', fontsize=12)
# plt.title(f'CRPS - Estación {EST}', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xticks(fh_temporal, rotation=45)
# #plt.legend(title='Configuración', fontsize=10)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.plot(np.arange(1,6), picp, '*-', color=color, linewidth=1, markersize=8)
# plt.xlabel('Horizonte temporal (min)', fontsize=12)
# plt.ylabel('picp', fontsize=12)
# plt.title(f'PICP - Estación {EST}', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xticks(fh_temporal, rotation=45)
# #plt.legend(title='Configuración', fontsize=10)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.plot(np.arange(1,6), pinaw, '*-', color=color, linewidth=1, markersize=8)
# plt.xlabel('Horizonte temporal (min)', fontsize=12)
# plt.ylabel('pinaw', fontsize=12)
# plt.title(f'PINAW - Estación {EST}', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xticks(fh_temporal, rotation=45)
# #plt.legend(title='Configuración', fontsize=10)
# plt.tight_layout()
# plt.show()    
    


#%%

# metricas_fh = []

# for fh in fh_temporal:
#         print(f'Trabajando con conjunto {conj_f_optima[fh]}')
#         Dataset = c_fn.dataset(path, fh, EST, year_train, year_test, lag)
#         X_train, X_test, y_train, y_test = Dataset.get_data(conj_f_optima[fh])

#         modelo = c_fn.Regresion_cuantil(kc_min, kc_max, quantiles)
#         modelo.fit(X_train, y_train)
#         X_pred = modelo.predict(X_test)
        
#         gcs = Dataset.target(Dataset.data_test['gcs'])
        
#         met = c_fn.metricas(X_pred, y_test, alpha, gcs)
#         metricas_fh.append(met.cal_metricas(modelo.quantiles))
        



# # fin = time.time()

# #%% GRAFICO DE LOS MODELOS CON DISTINTOS CONJUNTOS DE FEATURES

# #METRICAS: conf,PICP,ACE,PINAW,CRPS

# # PICP, PINAW y CRPS por variable
# picp = list(zip(*metricas_fh))[1]
# pinaw = list(zip(*metricas_fh))[3]
# crps = list(zip(*metricas_fh))[4]

# color = '#1f77b4'
        
# plt.figure(figsize=(8, 6))
# plt.plot(np.arange(1,6), crps, '*-', color=color, linewidth=1, markersize=8)
# plt.xlabel('Horizonte temporal (min)', fontsize=12)
# plt.ylabel('crps', fontsize=12)
# plt.title(f'CRPS - Estación {EST}', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xticks(fh_temporal, rotation=45)
# #plt.legend(title='Configuración', fontsize=10)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.plot(np.arange(1,6), picp, '*-', color=color, linewidth=1, markersize=8)
# plt.xlabel('Horizonte temporal (min)', fontsize=12)
# plt.ylabel('picp', fontsize=12)
# plt.title(f'PICP - Estación {EST}', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xticks(fh_temporal, rotation=45)
# #plt.legend(title='Configuración', fontsize=10)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.plot(np.arange(1,6), pinaw, '*-', color=color, linewidth=1, markersize=8)
# plt.xlabel('Horizonte temporal (min)', fontsize=12)
# plt.ylabel('pinaw', fontsize=12)
# plt.title(f'PINAW - Estación {EST}', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xticks(fh_temporal, rotation=45)
# #plt.legend(title='Configuración', fontsize=10)
# plt.tight_layout()
# plt.show()

    

# fin = time.time()
# print(f'Tiempo total de ejecución: {fin - inicio:.2f} segundos')




# c_features1 =  ['var','S6']


# metricas = []

# for fh in fh_temporal:
#     Dataset = c_fn.dataset(path,fh,EST,year_train,year_test,lag) 
    
#     X_train, X_test, y_train, y_test = Dataset.get_data(c_features1)

#     rcuantil = c_fn.Regresion_cuantil(kc_min, kc_max, quantiles)
    
#     rcuantil.fit(X_train,y_train)
#     X_predicha = rcuantil.predict(X_test)
    
#     metricas.append(c_fn.metricas(X_predicha,y_test,alpha,X_test['gcs'],quantiles))



# #%% GRÁFICAS de métricas 

# #DIAGRAMA DE CONFIABILIDAD
# plt.figure(figsize=(8, 6))  # tamaño más grande
# plt.plot(quantiles, np.arange(10, 100, 10), 'o-', color='black', label='Calibración Perfecta')
# colors = plt.cm.viridis(np.linspace(0, 1, len(fh_temporal)))  # mapa de colores

# for i, fh in enumerate(fh_temporal):
#     frec_obs = metricas[i][0]  # asumimos que el índice 0 contiene la curva de confiabilidad
#     plt.plot(quantiles, frec_obs, '--o', color=colors[i], label=f'{15*fh} minutos')
# plt.xlabel('Cuantil pronosticado', fontsize=12)
# plt.ylabel('Frecuencia observada', fontsize=12)
# plt.title(f'Diagrama de confiabilidad - Estación {EST}', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(title='Horizonte temporal', fontsize=10)
# plt.tight_layout()


# # --- GRAFICOS ---
# # PICP
# c_fn.plot_metric(fh_temporal, list(zip(*metricas))[1], 'PICP', 'PICP', c_features1, EST)

# # PINAW
# c_fn.plot_metric(fh_temporal, list(zip(*metricas))[3], 'PINAW', 'PINAW', c_features1, EST)

# # CRPS
# c_fn.plot_metric(fh_temporal, list(zip(*metricas))[4], 'CRPS', 'CRPS', c_features1, EST)


#%% INTENTO DE RED NEURONAL

'''import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

fh = 1 
n_epoch = 40 


Dataset = c_fn.dataset(path, fh, EST, year_train, year_test, lag)
gcs = Dataset.target(Dataset.data_test['gcs'])

X_train, X_test, y_train, y_test = Dataset.get_data(['var','S6'])


modelo = c_fn.RedNeuronal(20, int(X_train.size/len(X_train)), quantiles)

optimizer = optim.Adam(modelo.parameters())

dataset_radiation = c_fn.DatasetRadiacion(X_train,y_train)
dataloader = DataLoader(dataset_radiation, batch_size=250 ,shuffle=False)

modelo.fit(optimizer,n_epoch,quantiles,dataloader)

X_predich = modelo.predict(X_test)

metricas = c_fn.metricas(X_predich,y_test,0.2,gcs)

met = metricas.cal_metricas(quantiles)'''
    
    
    
    