# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:55:08 2025

@author: Victoria
"""

#14 DE ABRIL, CODIGO PARA UNIFICAR DATOS DE PCA CON DATOS DE LAS ESTACIONES

import pandas as pd

#definicion de variables 
pca_path = 'C:/Users/Victoria/OneDrive/Documents/Proyecto ANII/datos_sw_victoria/imagenes_satelitales/CAMS_LOC'
data_path = 'C:/Users/Victoria/OneDrive/Documents/Proyecto ANII/datos_sw_victoria/datasets_15min_art/datasets_15min_art/datasets_15min_art'
path_final= 'C:/Users/Victoria/OneDrive/Documents/Proyecto ANII/CODIGO VERSION 14 DE ABRIL'


year_train = 2017
year_test = 2018
estaciones = ['CAB','CAR','CEN','MIL','PAL','PAY','TAB','TOR']


def cargar_PCA(EST, year_train, year_test, pca_path):
    df_list = []
    for year in [year_train, year_test]:
        df = pd.read_csv(f"{pca_path}/PCA_{EST}_{year}.csv", parse_dates=['timestamp'])
        df_list.append(df)
    df_pca = pd.concat(df_list, ignore_index=True)
    return df_pca


def cargar_datos_estacion(EST,data_path):
    df = pd.read_csv(f"{data_path}/{EST}_e15_all.csv", parse_dates=['datetime'])
    df = df[df['msk'] == 1].reset_index(drop=True) #FILTRADO DE DATOS ASOCIADOS AL DIA 
    return df


def unir_datos(df_estacion, df_pca):
    df_merged = pd.concat([df_estacion, df_pca[df_pca.columns[1:]]], axis=1)
    return df_merged


for EST in estaciones:
    
    df_pca = cargar_PCA(EST, year_train, year_test, pca_path)
    df = cargar_datos_estacion(EST,data_path)
    
    df_merged = unir_datos(df, df_pca) 
    df_merged.to_csv(f'{path_final}/{EST}.csv', index=False) ##GUARDADO COMO CSV 



#%%
'''
VERIFICACION QUE EL VECTOR DE TIEMPO ES EL MISMO EN AMBOS data frames
lomismo = (df.datetime == df_pca.timestamp)

if(sum(lomismo)== len(df) and sum(lomismo)== len(df_pca)):
    print('EL VECTOR TIEMPO ES EL MISMO')
    
else:
    print('Hay un error en el vector tiempo')

'''



