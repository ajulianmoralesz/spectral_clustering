# -*- coding: utf-8 -*-
"""
@author: Arnold Julian Morales Zapata
"""

import numpy as np
import scipy.sparse as sps
import scipy.spatial as sp
import utilidades as util
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import math

datasets = ['DB/happy', 'DB/be3', 'DB/sp','DB/hm', 'DB/tar'] 
sigmas = [0.06, 0.06, 0.4, 0.1, 0.3 ]
clases = [3, 3, 3, 2, 6 ]


#Creacion de Grafos

def Crear_Grafo_CNN(data, epsilon):
    D =  sp.distance_matrix(data, data);
    CNN = np.asarray(np.double(D < epsilon));
    CNN = CNN * CNN.T;
    return CNN

def Crear_Grafo_KNN(data, K):
    D =  sp.distance_matrix(data, data);    
    r, c = D.shape    
    n_D = D + np.diag(np.diag(D) + math.inf);
    Grafo =  np.zeros((r,c));
    for i in range(r):
        f_a = n_D[i,:];
        f_or = np.sort(np.copy(f_a))[0:K];
        for j in range(K):
            for k in range(c):
                if D[i,k]==f_or[j]:
                    Grafo[i,k]=1;
                    break;    
    return Grafo;
    
#region Aplicacion de Kernel
def Kernel_Powered_Gaussian(data, c, m=1):
    """
    Spectral clustering with density sensitive similarity function
    Peng Yang - Qingsheng Zhu - Biao Huang
    Entradas
    data = matriz de caracteristicas
    c= Valor constate
    m= Valor incremental
    """
    D =  sp.distance_matrix(data, data);
    Y = D + np.diag(np.diag(D) + math.inf);
    Beta = np.max(np.amin(Y, axis=0));
    umbral = 0.003;
    gamma = c*m;
    f_c = umbral + 1;
    while f_c > umbral:
        J_sm = np.exp(-D/Beta)**gamma;
        J_sm1 = np.exp(-D/Beta)**(c*(m+1));
        f_c = np.mean(np.corrcoef(J_sm, J_sm1)); #Dudas en este punto, preguntar a Andres
        m+=1;
        gamma = c*m;
    kernel = np.exp(-D/Beta)**gamma;    
    return  kernel - np.diag(np.diag(kernel));
    
def Kernel_ALS(data, f_d, K=0):
    """
    Spectral clustering with density sensitive similarity function
    Peng Yang - Qingsheng Zhu - Biao Huang
    Entradas
    data = matriz de caracteristicas
    f_d = factor de densidad
    K = numero de vecinos para construir [Opcional]
    Se recomienda usar el valor por cuestiones de costo computacional
    """
    grafo = util.Crear_Grafo(data, K);
    dist2 = np.zeros(grafo.shape);
    Dp = (np.exp(f_d*grafo)-1)**(1/f_d);    
    for i in range(data.shape[0]):
        dist2[i,:]=util.Grafo_Ruta_Mas_Corta(Dp,i);#Dudas en este punto con lo que devuelve el codigo suministrado en Matlab     
    kernel = 1/(dist2+1);
    kernel =  (kernel + kernel.T)/2;
    return kernel - np.diag(np.diag(kernel));


def Kernel_CNN(data, sigma, epsilon):            
    """
    Local density adaptive similarity measurement for spectral clustering
    Xianchao Zhang, Jingwei Li, Hong Yu
    Entradas
    data = Matriz de caracteristicas
    sigma = Ancho de banda del kernel  
    epsilon = valor del vecindario para cada dato.
    """
    D =  sp.distance_matrix(data, data);
    CNN = np.asarray(np.double(D < epsilon));#Esto es el epsilon nn
    CNN = CNN * CNN.T;
    kernel = np.exp(-(D**2)/(2*(sigma**2)*(CNN+1)));
    return kernel - np.diag(np.diag(kernel));

def Kernel_Local_Scaling(data, K_vecino):
    """
    Self-Tuning Spectral Clustering
    Lihi Zelnik-Manor - Pietro Perona     
    Entradas
    data = Matriz de caracteristicas
    K_vecino = k-esimo vecino. [Segun Zelnik igual a 7]
    [Con pruebas realizadas, K_vecino > 15 presenta fallos, valores menores funcionan]
    """        
    distx= np.sort(sp.distance.squareform(sp.distance.pdist(data)), axis=0)    
    n = np.asarray((data)).shape[0];
    Dif = np.zeros((n, n));    
    sig1 = np.asarray(distx[K_vecino,:]).reshape((n,1));     
    for i in range(n):    
        tmp = np.matlib.repmat(data[i,:],n,1)
        tmp[i][0]= math.inf;
        Dif[i,:] = np.sum((tmp-data)**2, axis=1)        
    kernel = np.exp(-Dif/(np.multiply(sig1, np.asarray(sig1).T)));
    return kernel - np.diag(np.diag(kernel)); 


def Kernel(data, sigma):
    """
    On Spectral Clustering: Analisys and an algorithm
    Andrew Y. Ng - Michael I. Jordan - Yair Weiss
    Entradas
    data = Matriz de caracteristicas
    sigma = Ancho de banda del kernel
    """
    m_dis = sp.distance_matrix(data, data)
    A = np.exp(-m_dis/(2.0*sigma**2.0))    
    return A

def Laplaciano_No_Normalizado(M):
    """
    J. Shi and J. Malik,
    Calcula el laplaciano normalizado simétrico
    L = D-M
    """    
    w = np.sum(M, axis=0)**(-0.5);
    D=np.diag(w);
    return D-M

def Laplaciano_Normalizado(M):
    """
    Ng, A., Jordan, M., and Weiss, Y. (2002)
    Calcula el laplaciano normalizado simétrico
    L = D^{-1/2} A D{-1/2}    
    
    """    
    w = np.sum(M, axis=0)**(-0.5);
    D=np.diag(w);
    return D.dot(M).dot(D);

def Eig_Vects(Laplaciano, K):
    val_prop, vect_prop = sps.linalg.eigs(Laplaciano, K)
    X = vect_prop.real    
    rows_norm = np.linalg.norm(vect_prop.real, axis=1, ord=2)
    return (X.T / rows_norm).T

def k_means(X, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=1231)
    return km.fit(X).labels_ #.fit(X).labels_

def spectral_demo(seleccion):  
    if seleccion>4:
        print('No es una opción valida [0-4]')
        return
    C=clases[seleccion]
    datos= util.cargar_mat('DB.mat', datasets[seleccion]) #0 v 8
    f, axarr = plt.subplots(5, 2); 
    #Kernel Base
    AF = Kernel(datos, 0.07);
    axarr[0, 0].imshow(AF);
    axarr[0, 0].set_title('Kernel Base Sigma=0.07');    
    Lap = Laplaciano_Normalizado(AF)    
    Y = Eig_Vects(Lap, C);
    etiquetas = k_means(Y, C);   
    axarr[0, 1].scatter(datos[:,0], datos[:,1],c= etiquetas)
    axarr[0, 1].set_title('Resultado Kernel Base');
    #kernel localscaling
    AF = Kernel_Local_Scaling(datos, 5);    
    axarr[1, 0].imshow(AF);
    axarr[1, 0].set_title('Kernel Local Scaling K=5');
    Lap = Laplaciano_Normalizado(AF);
    Y = Eig_Vects(Lap, C);
    etiquetas = k_means(Y, C);   
    axarr[1, 1].scatter(datos[:,0], datos[:,1],c= etiquetas)
    axarr[1, 1].set_title('Resultado Local Scaling');
    #Kernel CNN
    AF = Kernel_CNN(datos,0.03, 0.05);    
    axarr[2, 0].imshow(AF);
    axarr[2, 0].set_title('Kernel CNN Sigma=0.03, Epsilon=0.05');
    Lap = Laplaciano_Normalizado(AF);
    Y = Eig_Vects(Lap, C);
    etiquetas = k_means(Y, C);   
    axarr[2, 1].scatter(datos[:,0], datos[:,1],c= etiquetas)
    axarr[2, 1].set_title('Resultado CNN');
    #Kernel Powered Gaussian
    AF = Kernel_Powered_Gaussian(datos,0.7);    
    axarr[3, 0].imshow(AF);
    axarr[3, 0].set_title('Kernel PG c=0.7');
    Lap = Laplaciano_Normalizado(AF);
    Y = Eig_Vects(Lap, C);
    etiquetas = k_means(Y, C);   
    axarr[3, 1].scatter(datos[:,0], datos[:,1],c= etiquetas)
    axarr[3, 1].set_title('Resultado PG');
    #Kernel ALS
    AF = Kernel_ALS(datos,4, 0);    
    axarr[4, 0].imshow(AF);
    axarr[4, 0].set_title('Kernel ALS Fd=8, K=0');
    Lap = Laplaciano_Normalizado(AF);
    Y = Eig_Vects(Lap, C);
    etiquetas = k_means(Y, C);   
    axarr[4, 1].scatter(datos[:,0], datos[:,1],c=etiquetas)
    axarr[4, 1].set_title('Resultado ALS');    
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)    
    plt.show()
    return
    
spectral_demo(0);



