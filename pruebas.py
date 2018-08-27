# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 09:15:50 2018

@author: analista1
"""

import numpy as np
import utilidades as util
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.spatial as sp
import scipy.sparse as sps
import networkx as nx
import math

datasets = ['DB/happy', 'DB/be3', 'DB/sp','DB/hm', 'DB/tar'] 
sigmas = [0.06, 0.06, 0.4, 0.1, 0.3 ]
clases = [3, 3, 3, 2, 6 ]
seleccion = 0
data= util.cargar_mat('DB.mat', datasets[seleccion]);



D =  sp.distance_matrix(data, data);
n_D = D + np.diag(np.diag(D) + math.inf);
r, c = D.shape;
Grafo =  np.zeros((r,c));
for i in range(r):
    f_a = n_D[i,:];
    f_or = np.sort(np.copy(f_a))[0:3];
    for j in range(3):
        for k in range(c):
            if D[i,k]==f_or[j]:
                Grafo[i,k]=1;
                break;







