# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 07:44:15 2018

@author: analista1
"""

import numpy as np
import scipy.io
import h5py
import scipy.spatial as sp
import scipy.sparse as sps
import math
import networkx as nx


def BFS(mat_adj, raiz):
    """
    breadth first search
    Amplitud de primera busqueda
    Entrada : Matriz de adyajencia, sobre la cual se va a cear el grafo
    raiz = numero de vertice sobre el cual se va a iniciar
    """
    rows, cols = np.where(mat_adj > 0)
    if rows.shape[0]<=0 and cols.shape[0]<=0:
        return np.asarray([]);
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    #return np.asarray(list(nx.bfs_edges(gr, 0)));
    return np.asarray(list(nx.bfs_tree(gr, raiz)));    

def dist(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return ((a - b) ** 2)  ** 0.5


def Grafo_Ruta_Mas_Corta(mat_adj, raiz):
    row, col = mat_adj.shape;
    rows, cols = np.where(mat_adj > 0)    
    if rows.shape[0]<=0 and cols.shape[0]<=0:
        return np.asarray([]);
    edges = zip(rows.tolist(), cols.tolist())    
    gr = nx.Graph()
    gr.add_edges_from(edges)        
    ed = np.asarray(list(gr.edges()))
    ruta = []
    for i in range(row): 
        if i == raiz:
            ruta.append(0);
        else:
            if raiz > i:
                path =nx.shortest_path(gr, i, raiz);
            else:
                path =nx.shortest_path(gr, raiz, i);
            
            m_v =[];            
            if len(path)==2:
                c = np.asarray(ed[path[1]-1]);
                m_v.append(mat_adj[c[0], c[1]]);
            else:   
                for ind in path:                  
                    c = np.asarray(ed[ind]);
                    m_v.append(mat_adj[c[0], c[1]]);
            if np.asarray(m_v).shape[0]==1:
                ruta.append(m_v[0]);
            else:
                val = np.linalg.norm([m_v])
                ruta.append(val);        
    return ruta;

def cargar_mat(ruta, nombre_db):
    try:
        mat = scipy.io.loadmat(ruta)
    except NotImplementedError:
        mat = h5py.File(ruta)
    print(mat)
    return np.array(mat[nombre_db]).transpose()

def Crear_Grafo(data, K=0):
    """     
    Entradas
    data = Matriz de datos
    K_vecino = k-esimo vecino.
    Salida
    Matriz con los pesos de las aristas
    """
    D = sp.distance_matrix(data, data);
    D = D + np.diag((math.inf * np.diag(np.ones(D.shape))));
    ind=np.argsort(D, axis=0); #obtiene el array de indices en el cual se ordeno    
    D_s = np.copy(D);
    if K==0:
        s = D_s.shape;        
        for K in range(s[0]):            
            D_s = np.copy(D);
            for i in range(s[0]):
                D_s[i, ind[K:s[0],i]]= math.inf;                
            D_s = np.minimum(D_s,D_s.T);
            #conectividad en el grafo
            E = 1.0-np.double(D_s==math.inf);
            E=E*np.double(np.eye(s[0])==0);
            va = False; 
            if np.sum(E) > 0:                           
                for j in range(s[0]):#range(s[0]):                    
                    bfs = BFS(E,j);
                    if bfs.shape[0] < s[0]: #verificando que todo este conectado
                        break;
                    if j==s[0]-1:
                        va = True;
            if va:
                break;                      
    else:        
        for i in range(D_s.shape[0]):
            D_s[i, ind[K-1:D_s.shape[0],i]]=math.inf;
        D_s = np.minimum(D_s,D_s.T);
    D_s[np.isinf(D_s)] = 0
    return D_s

