# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:09:40 2020

@author: carlosa-correaf
"""

import numpy as np


Curvapu=0.01*np.array([60,65,70,70,75,80,80,80,80,80,75,75,75,80,85,90,90,95,100,100,95,85,80,65])

fch=2
Hx=len(np.where((Curvapu>=0.95))[0])
Hz=len(np.where((Curvapu<0.95) & (Curvapu>=0.75))[0])
Hy=len(np.where((Curvapu<0.75))[0])
Px=np.mean(Curvapu[np.where((Curvapu>=0.95))[0]])
Pz=np.mean(Curvapu[np.where((Curvapu<0.95) & (Curvapu>=0.75))[0]])
Py=np.mean(Curvapu[np.where((Curvapu<0.75))[0]])

Dt=350
sumPi=np.sum(Curvapu)

#construir la matriz

A=np.array([[Hx*Px/fch,Hz*Pz,fch*Hy*Py],[1/fch,-Px/Pz,0],[1/(fch*fch),0,-Px/Py]])
b=np.array([[Dt*sumPi],[0],[0]])

x = np.linalg.solve(A, b)

CostoVariable=np.sum(x[0]*Curvapu[np.where((Curvapu>=0.95))[0]])+np.sum(x[1]*Curvapu[np.where((Curvapu<0.95) & (Curvapu>=0.75))[0]])+np.sum(x[2]*Curvapu[np.where((Curvapu<0.75))[0]])
CostoFija=np.sum(Dt*Curvapu)
print(Curvapu)
print(CostoVariable)
print(CostoFija)