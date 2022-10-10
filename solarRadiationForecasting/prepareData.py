
# Reads data from IDEAM hydrometeorological databases for different variables
# In[1]:

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (20, 15)
mpl.rcParams['axes.grid'] = False

# In[2]:
def readVarData(path, fileName, variableName):
	data = pd.read_csv(path + fileName, index_col=['Etiqueta'])
	data = data[['Fecha', 'Valor']]
	data = data.filter(like=variableName, axis=0)
	data.reset_index(inplace=True)
	data.set_index('Fecha', inplace=True)
	data = data[['Valor']]
	data = data.rename(columns={'Valor': variableName})
	return data

# In[3]:
csv_path = '/home/gbecerra/Javeriana/Research/ProyectoAlmacenamiento/ideam/'
# Reads the solar radiation dataset
data_radsol = readVarData(csv_path,'radiacionSolar.csv', 'RSG_AUT_60')
# Reads the temperature dataset
data_temp = readVarData(csv_path,'temperatura.csv', 'TA2_2_MEDIA')
# Reads the relative humidity dataset
data_humrel = readVarData(csv_path,'humedadRelativa.csv', 'HRA2_2_MEDIA_H')
# Reads the prepitation dataset
data_precip = readVarData(csv_path,'precipitacion.csv', 'PT_2_TT_H')
# reads the atmospheric pressure dataset
data_presatm = readVarData(csv_path,'presionAtmosferica.csv', 'PA_AUT_60')
# reads the wind speed dataset
data_velviento = readVarData(csv_path,'velocidadViento.csv', 'VV_2_MEDIA_H')
# reads the wind direction dataset
data_dirviento = readVarData(csv_path,'direccionViento.csv', 'DV_2_VECT_MEDIA_H')

# In[4]:
data_tot = data_radsol
# data_tot = data_radsol.join(data_temp,how='outer')
data_tot = data_tot.join(data_humrel,how='outer')
data_tot = data_tot.join(data_precip,how='outer')
# data_tot = data_tot.join(data_presatm,how='outer')
data_tot = data_tot.join(data_velviento,how='outer')
data_tot = data_tot.join(data_dirviento,how='outer')
# data_tot = data_tot.dropna(thresh=4)
# data_tot

# In[5]:
# Creates missing data by forward-filling
for row in range(data_tot.shape[0]):
	for col in range(data_tot.shape[1]):
		if pd.isna(data_tot.iloc[row,col]):
			if row < 24:
				data_tot.iloc[row,col] = 0
			else:
				data_tot.iloc[row,col] = data_tot.iloc[row-24,col]

# In[6]:
data_tot.plot(subplots=True)

# In[7]:
# Saves the dataframe to a csv file
data_tot.to_csv(csv_path + 'data_tot.csv')