import networkx as nx
import numpy as np
import warnings
import itertools
from tqdm import tqdm_notebook
from scipy.special import perm, factorial
import os


def simplex2hasse_uniform(data, max_order=None):
    '''Returns the Hasse diagram with "unweighted" weighting scheme as a networkX graph (undirected, simple). 
    Input: list of frozensets
    Output: networkx Graph object
    '''
    
    def _build_simplices(simplex, l):
        #recursive function that calculates all possible simplices
        for s in itertools.combinations(simplex, len(simplex)-1):
            if len(s) > 0:
                l.add((frozenset(simplex),frozenset(s)))
            
            if len(s)>1:
                _build_simplices(s,l)
        return 


    # execute cleaning of the dataset - remove duplicate simplices
    #data = list(set(data))

    # initialize the Hasse graph (diagram)
    g = nx.Graph()

    # go through the simplices, create nodes and edges
    for u in tqdm_notebook(data, 'Creating Hasse diagram'):
        if None == max_order or (len(u)<max_order+1 and len(u) >= 1):
            buff = set({})
            _build_simplices(u, buff)
            g.add_edges_from(buff)
        else:
            for v in itertools.combinations(u,max_order+1):
                buff = set({})
                _build_simplices(v, buff)
                g.add_edges_from(buff)
    
    weight = 1.
    nx.set_node_attributes(g, weight,'weight')
    
    return g


def simplex2hasse_counts(data, max_order=None):
    '''Returns the Hasse diagram with "counts" weighting scheme as a networkX graph (undirected, weighted). Simplices appearing in the dataset receive the non-trivial weight that equals to the number of their appearance.
    Input: list of frozensets
    Output: networkx Graph object
    '''

    epsilon = 1.
       
    def _build_simplices(simplex, l):
        #recursive function that calculates all possible son simplices
        for s in itertools.combinations(simplex, len(simplex)-1):
            fs = frozenset(s)
            if len(s)>0:
                if fs not in weights_dict:
                    weights_dict[fs] = epsilon
#                 else:
#                     weights_dict[fs] += epsilon
                l.add((frozenset(simplex), fs))
            if len(s)>1:
                _build_simplices(s,l)
        return  
    
    
    # execute cleaning of the dataset - remove duplicate simplices
    #data = list(set(data))

    # initialize the Hasse graph (diagram)
    g = nx.Graph()
    weights_dict = {}
    
    # go through the simplices, create nodes
    for u in tqdm_notebook(data, 'Creating Hasse diagram'):
        if None == max_order or (len(u) < max_order+1 and len(u) >= 1):
            if u not in weights_dict:
                weights_dict[u] = 1.
            else:
                weights_dict[u] += 1.
            buff = set({})
            _build_simplices(u, buff)
            g.add_edges_from(buff)
        else:
            for v in itertools.combinations(u, max_order+1):
                if frozenset(v) not in weights_dict:
                    weights_dict[frozenset(v)] = 1.
                else:
                    weights_dict[frozenset(v)] += 1.
                buff = set({})
                _build_simplices(v, buff)
                g.add_edges_from(buff)
    
    nx.set_node_attributes(g, weights_dict, 'weight')
    
    return g

def simplex2hasse_LOexponential(data, max_order=None):
    '''Returns the Hasse diagram with "lobias" weighting scheme as a networkX graph (undirected, weighted) with cumulative appearance counts on nodes adjusted by the diagram level. Adjustment coefficient for n-simplex on level k (level of k-simplices) is (n+1)*n*..*(n-k+1)
    
    Example: 3-simplex (tetrahedron) appearing in the data receives weight 1, adjacent 2-simplices (triangles) receive weigth 4, 1-simplices (edges) receive (4*3), 0-simplices (nodes) receive (4*3*2).
    Input: list of frozensets
    Output: networkx Graph object
    '''
        
    def _build_simplices(simplex, l):
        #recursive function that calculates all possible son simplices
        for s in itertools.combinations(simplex, len(simplex)-1):
            fs = frozenset(s)
            #top_simplex_order = max_order
            level = top_simplex_order - len(simplex) + 1
            if len(s) > 0:
                
                if fs in weights_dict:
                    weights_dict[fs] += factorial(top_simplex_order)/factorial(top_simplex_order-level)
                else:
                    weights_dict[fs] = factorial(top_simplex_order)/factorial(top_simplex_order-level)

                l.add((frozenset(simplex), fs))
            if len(s) > 1:
                _build_simplices(s,l)
        return  
    
    
    # execute cleaning of the dataset - remove duplicate simplices
    #data = list(set(data))

    # initialize the Hasse graph (diagram)
    g = nx.Graph()
    weights_dict = {}
    
    # go through the simplices, create nodes
    for u in tqdm_notebook(data, 'Creating Hasse diagram'):

        if None == max_order or (len(u) < max_order+1 and len(u) >= 1):
            if u not in weights_dict:
                weights_dict[u] = 1.
            else:
                weights_dict[u] += 1.
            buff = set({})
            top_simplex_order = len(u)
            _build_simplices(u, buff)
            g.add_edges_from(buff)
        else:
            for v in itertools.combinations(u, max_order+1):
                if frozenset(v) not in weights_dict:
                    weights_dict[frozenset(v)] = 1.
                else:
                    weights_dict[frozenset(v)] += 1.
                buff = set({})
                top_simplex_order = len(v)
                _build_simplices(v, buff)
                g.add_edges_from(buff)

    nx.set_node_attributes(g, weights_dict, 'weight')
    
    return g


def simplex2hasse_HOexponential(data, max_order=None):
    '''Returns the Hasse diagram with "hobias" weighting scheme as as a networkX graph (undirected, weighted) with cumulative appearance counts on nodes adjusted by the diagram level. Adjustment coefficient for n-simplex on level k (level of k-simplices) is 1/((n+1)*n*..*(n-k+1))
    
    Example: 3-simplex (tetrahedron) appearing in the data receives weight 1, adjacent 2-simplices (triangles) receive weigth 1/4, 1-simplices (edges) receive 1/(4*3), 0-simplices (nodes) receive 1/(4*3*2).
    Input: list of frozensets
    Output: networkx Graph object
    '''
        
    def _build_simplices(simplex, l):
        #recursive function that calculates all possible son simplices
        for s in itertools.combinations(simplex, len(simplex)-1):
            fs = frozenset(s)
            #top_simplex_order = max_order
            level = top_simplex_order - len(simplex) + 1
            if len(s) > 0:
                
                if fs in weights_dict:
                    weights_dict[fs] += factorial(top_simplex_order-level)/factorial(top_simplex_order)
                else:
                    weights_dict[fs] = factorial(top_simplex_order-level)/factorial(top_simplex_order)

                l.add((frozenset(simplex), fs))
            if len(s) > 1:
                _build_simplices(s,l)
        return  
    
    
    # execute cleaning of the dataset - remove duplicate simplices
    #data = list(set(data))

    # initialize the Hasse graph (diagram)
    g = nx.Graph()
    weights_dict = {}
    
    # go through the simplices, create nodes
    for u in tqdm_notebook(data, 'Creating Hasse diagram'):

        if None == max_order or (len(u) < max_order+1 and len(u) >= 1):
            if u not in weights_dict:
                weights_dict[u] = 1.
            else:
                weights_dict[u] += 1.
            buff = set({})
            top_simplex_order = len(u)
            _build_simplices(u, buff)
            g.add_edges_from(buff)
        else:
            for v in itertools.combinations(u, max_order+1):
                if frozenset(v) not in weights_dict:
                    weights_dict[frozenset(v)] = 1.
                else:
                    weights_dict[frozenset(v)] += 1.
                buff = set({})
                top_simplex_order = len(v)
                _build_simplices(frozenset(v), buff)
                g.add_edges_from(buff)

    
    nx.set_node_attributes(g, weights_dict, 'weight')
    
    return g