# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:10:27 2025

@author: Victoria
"""

import pandas as pd 
import numpy as np
from sklearn.linear_model import QuantileRegressor
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
from torch.utils.data import Dataset,DataLoader


#CLASE PARA GENERAR LA MATRIZ CON LAS VARIABLES PREDICTORAS o features
class dataset:
    def __init__(self,path_data,fh,EST,year_train,year_test,lag): #metodo constructor
        
        self.path_data = path_data
        self.year_train = year_train 
        self.year_test = year_test
        self.EST = EST 
        self.lag = lag
        self.fh = fh
        
        self.data = pd.read_csv(f'{self.path_data}/{self.EST}.csv')
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        
        self.data_train = self.data[self.data['datetime'].dt.year == self.year_train] #MATRIZ DE DATOS TRAIN
        self.data_test = self.data[self.data['datetime'].dt.year == self.year_test] #MATRIZ DE DATOS TEST

    
    def conj_features(self,data,lista_variables): 
        df = pd.DataFrame()
        
        df['kc'] = data['kc']
        columnas = []
      
        # Añadir las columnas de lags de 'kc'
        for i in range(1, self.lag + 1):
            df[f'kc_t-{i}'] = data['kc'].shift(i)
            columnas.append(f'kc_t-{i}')
            
        for var in lista_variables:
            if var in data.columns:
                df[var] = data[var]
            else:
                print(f"Advertencia: la variable '{var}' no está en el DataFrame original.")
        
        df = df.dropna().reset_index(drop=True) #Eliminar NaNs generados por los lags
        df = df[:-self.fh] #Eliminar datos futuros 
        
        #df['gcs'] = self.target(data['gcs']) #AGREGA LA COLUMNA DE GCS 
        
        return df
        
    
    def target(self,x): #RETORNA EL VECTOR x CORRIDO 
        y = x.shift(-(self.lag + self.fh))
        y = y[:-(self.fh + self.lag)]
        y = y.reset_index(drop=True)
        
                
        return y
    
    def get_data(self, lista_variables):
        X_train = self.conj_features(self.data_train, lista_variables)
        X_test = self.conj_features(self.data_test, lista_variables)
        
        y_train = self.target(self.data_train['kc'])
        y_test = self.target(self.data_test['kc'])
        
        return X_train, X_test, y_train, y_test
    
    
class metricas():
    def __init__(self,y_pred,y,alfa,gcs):
        self.y_pred = y_pred
        self.y = y
        self.alfa = alfa
        self.gcs = gcs
        
        self.gcs = self.gcs.values.reshape(-1, 1)


        if isinstance(self.y, pd.Series):
            self.y = self.y.values.reshape(-1, 1)
        elif isinstance(self.y, pd.DataFrame):
            self.y = self.y.values.reshape(-1, 1)
        elif isinstance(self.y, np.ndarray):
            self.y = self.y.reshape(-1, 1)
        else:
            raise TypeError("El vector y debe ser un Series, DataFrame o ndarray.")
 
    def calcular_CDF_x(self, x, xvec):

        N = x.size
        s = xvec.size
        Fx = np.zeros((s))
    
        for k in range(N):
            Fx = Fx + (xvec >= x[k])
        
        Fx = Fx / N
    
        return Fx


    def CRPS_metric_B(self, Qx, qs, yval):

        CRPS = 0
        N = len(yval)
    
        if isinstance(yval, (np.ndarray, list)):
            yval = pd.Series(yval)
    
        for j in range(N):
            xvec = np.linspace(0, Qx.iloc[j, :].max(), 100)
            dx = xvec[1] - xvec[0]
    
            CDFj = np.interp(xvec, Qx.iloc[j, :], qs)
            CDFm = self.calcular_CDF_x(np.array([yval.iloc[j]]), xvec)
    
            CRPS += dx * np.sum((CDFj - CDFm) ** 2)
    
        return CRPS / N


    def cal_metricas(self,quantiles):
    
      
            
        y_pred_ghi = self.y_pred * self.gcs  
        y_ghi = self.y * self.gcs
    
        y_pred_ghi = pd.DataFrame(y_pred_ghi)
        y_ghi = pd.Series(y_ghi.squeeze())  
        
    
        conf = (y_pred_ghi >= y_ghi.values.reshape(-1, 1)).mean() * 100
    
        LI = self.alfa / 2
        LS = 1 - LI
    
        # Paso 5: Encontrar índices de los cuantiles más cercanos a LI y LS
        i_LI = np.where(np.isclose(quantiles, LI))[0]
        i_LS = np.where(np.isclose(quantiles, LS))[0]
    
        if len(i_LI) == 0 or len(i_LS) == 0:
            raise ValueError(f"No se encontraron los cuantiles {LI} o {LS} en self.quantiles.")
    
        q_LI = y_pred_ghi.iloc[:, i_LI[0]]
        q_LS = y_pred_ghi.iloc[:, i_LS[0]]
    
        PICP = np.mean(((q_LI <= y_ghi) & (y_ghi <= q_LS)).astype(int)) * 100
        
        
        ACE  = PICP  - 100*(1-self.alfa) 
        
        
        PINAW = ((q_LS - q_LI).sum() / y_ghi.sum())*100
        
        CRPS = self.CRPS_metric_B(y_pred_ghi, quantiles,y_ghi)
        
        
    
        return conf,PICP,ACE,PINAW,CRPS
    
def mejores_variables_por_fh(df_metricas):
    tipos = ['S', 'B', 'C', 'PCA']
    mejores_por_fh = {}

    horizontes = sorted(df_metricas['FH'].unique())

    for fh in horizontes:
        mejores_fh = []
        df_fh = df_metricas[df_metricas['FH'] == fh]
        
        for tipo in tipos:
            df_tipo = df_fh[df_fh['Variable'].str.startswith(tipo)]
            if not df_tipo.empty:
                mejor_variable = df_tipo.loc[df_tipo['CRPS'].idxmin(), 'Variable']
                mejores_fh.append(mejor_variable)
            else:
                mejores_fh.append(None)  

        mejores_por_fh[fh] = tuple(mejores_fh)  

    return mejores_por_fh


def feature_opt_fh(df_metricas,fh_temporal):
    
    idx_min = df_metricas.groupby('FH')['CRPS'].idxmin()
    
    features_opt = []
    
    for fh in fh_temporal:
        
        features_opt.append(df_metricas.loc[idx_min, ['FH', 'Variable', 'CRPS']].iloc[fh - 1]['Variable'])

    return features_opt

class Modelo(): 
    def __init__(self,kc_min, kc_max, quantiles):
        self.quantiles = quantiles
        self.kc_min = kc_min
        self.kc_max = kc_max 
    
    def fit(self,X,y): 
        '''
        Parameters
        ----------
        X : Matriz predictora
        y : vector target

        Returns
        -------
        None.
        '''
        pass
    def predict(self,X):
        '''
        Parameters
        ----------
        X : Matriz de evaluacion.

        Returns
        -------
        None.

        '''
        pass


    
#Clase regresion cuantil, hereda los atributos y metodos de Modelo 
class Regresion_cuantil(Modelo):
    def __init__(self,kc_min,kc_max,quantiles):
        super().__init__(kc_min,kc_max,quantiles)
    
    def fit(self,X,y):
        regresion = []
        for quantil in self.quantiles:
            regressor = QuantileRegressor(quantile=quantil, fit_intercept=True, alpha=0, solver='highs')
            regressor.fit(X, y)
            regresion.append(regressor)
        
        self.regresion = regresion
        
    def predict(self,X):
        X_test = []
        
        for regressor in self.regresion:
            predictor = regressor.predict(X)
            predictor = np.array(predictor)
            
            predictor[predictor < self.kc_min] = self.kc_min
            predictor[predictor > self.kc_max] = self.kc_max
            
            X_test.append(predictor)
        
        X_test = np.array(X_test)
        
        X_t= X_test.T #Transponer la matriz 
        X_t = np.sort(X_t, axis=1) #Reordenar por filas 
        
        return X_t
        
    

class DatasetRadiacion(Dataset):

    def __init__(self,dataframe,target):
        super().__init__()
        
        dataframe = dataframe.copy()  # Hacemos una copia para no modificar el original

        if 'target' in dataframe.columns:
            dataframe = dataframe.drop(columns=['target'])
        
        dataframe.insert(0, 'target', target)
        self.data = dataframe.to_numpy()

        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        features = self.data[idx, :-1]
        
        label = self.data[idx,-1]
        return features, label


class ArqRN(nn.Module): #ARQUITECTURA DE LA RED NEURONAL
    def __init__(self,dim_features,dim_quantiles,cant_neuronas):
        super().__init__()

        self.fc1 = nn.Linear(dim_features,cant_neuronas)
        self.fc2 = nn.Linear(cant_neuronas,dim_quantiles)
        
    def forward(self, x):

        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x 
    

class RedNeuronal(nn.Module,Modelo):
    def __init__(self,cant_neuronas,dim_features,quantiles):

        nn.Module.__init__(self)
        
        self.cant_neuronas = cant_neuronas
        self.dim_features = dim_features
        self.dim_quantiles = len(quantiles)
        
        self.modelo = ArqRN(dim_features,self.dim_quantiles,cant_neuronas)
        
    def forward(self,x):
        return self.modelo(x)
    
    def loss_pinball(self,y_true,y_pred,cuantil):
        
          diff = y_true - y_pred
          sign = (diff>=0).to(torch.int)
          loss = cuantil*(sign*diff) - (1-cuantil)*(1-sign)*diff    
          return loss.mean()
     
    def fit(self,optimizer,n_epoch,quantiles,dataloader):
          self.n_epoch = n_epoch
        
          #loss_for_epoch =[]
         
         
          for i in range(n_epoch):
              #print(i)
              for data in dataloader:
              # Set the gradients to zero
                  optimizer.zero_grad()
              # Run a forward pass
                  feature, target = data
                  prediction = self(feature.float()) 
                  loss = 0
              #loss = loss_pinball(target,prediction, quantil)   
             
                  for i, tau in enumerate(quantiles):
              # Calculate the loss
                  #print(tau)
                      loss += self.loss_pinball(target,prediction[:,i], tau)    
              # # Compute the gradients
                  loss/=len(quantiles)
             
             
              #print(loss)
              #loss = lossQuantil(prediction,target)
                  loss.backward()
              # Update the model's parameters
                  optimizer.step()
        

              #loss_for_epoch.append(loss.item())
         
          #return loss_for_epoch
          
    def predict(self,X):
        
        X = torch.from_numpy(X.values).float()
        
        self.modelo.eval()  
        
        with torch.no_grad():  
    
                #outputs = self.modelo(X)
                outputs = self.modelo(X.float()) 
    
        return outputs.numpy()  


# Colores consistentes
color = '#1f77b4'
path= 'C:/Users/Victoria/OneDrive/Documents/Proyecto ANII/CODIGO VERSION 14 DE ABRIL'


#FUNCIONES AUXILIARES PARA GRAFICOS 
def plot_metric(fh_temporal, y_values, ylabel, title, c_features1, EST, color=color):
    plt.figure(figsize=(8, 6))
    plt.plot(fh_temporal, y_values, '*-', color=color, linewidth=1, markersize=8, label=c_features1)
    plt.xlabel('Horizonte temporal (min)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{title} - Estación {EST}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fh_temporal, rotation=45)
    plt.legend(title='Configuración', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_fam_metrica(df_metricas, metrica_nombre, prefix, titulo=None):
    plt.figure(figsize=(10,6))
    subset = df_metricas[df_metricas['Variable'].str.startswith(prefix)]

    variables = sorted(subset['Variable'].unique(), key=lambda x: int(x[len(prefix):]))

    colors = plt.cm.viridis(np.linspace(0, 1, len(variables)))

    for var, color in zip(variables, colors):
        datos = subset[subset['Variable'] == var]
        plt.plot(datos['FH'], datos[metrica_nombre], label=var, color=color, marker='o')

    plt.xlabel('Horizonte temporal (FH)', fontsize=12)
    plt.ylabel(metrica_nombre, fontsize=12)
    plt.title(titulo or f'{metrica_nombre} para variables tipo {prefix}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Variable', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
def plot_graficas(fh_temporal,y,metrica_nombre, lista_var, titulo=None):    
    color = '#1f77b4'


    plt.plot(fh_temporal, y, label= lista_var, color=color, marker='o')

    plt.xlabel('Horizonte temporal (FH)', fontsize=12)
    plt.ylabel(metrica_nombre, fontsize=12)
    #plt.title(titulo or f'{metrica_nombre} para distintos', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Variable', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    
def plot_diag_conf(EST,quantiles,fh_temporal,metrica,variable):
    plt.figure(figsize=(8, 6))  # tamaño más grande
    plt.plot(quantiles, np.arange(10, 100, 10), 'o-', color='black', label='Calibración Perfecta')
    colors = plt.cm.viridis(np.linspace(0, 1, len(fh_temporal)))  # mapa de colores
    
    for i, fh in enumerate(fh_temporal):
        frec_obs = metrica[i][0]  # asumimos que el índice 0 contiene la curva de confiabilidad
        plt.plot(quantiles, frec_obs, '--o', color=colors[i], label=f'{15*fh} minutos')
    plt.xlabel('Cuantil pronosticado', fontsize=12)
    plt.ylabel('Frecuencia observada', fontsize=12)
    plt.title(f'Diagrama de confiabilidad - Estación {EST} - conjunto: {variable}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Horizonte temporal', fontsize=10)
    plt.savefig(f'{path}/graficos/confiabilidad_{EST}_{variable}.png')
    plt.tight_layout()

