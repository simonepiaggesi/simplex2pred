from gem.embedding.node2vec import node2vec
from subprocess import call
import tempfile
from gem.utils import graph_util
from time import time
import sys,os
import gzip

HOME = os.path.expanduser("~")

def loadWalks(file_name):
    walks = []
    with open(file_name, 'r') as f:
        for line in f:
            walk = list(map(int,line.strip().split()))
            walks.append(walk)
    return walks

class snap_node2vec(node2vec):
    def learn_embedding(self, graph=None, edge_f=None,
                        is_weighted=False, no_python=False, directed=False):
        args = [HOME+"/snap/examples/node2vec/node2vec"]
        if directed == False:
            graph_mod = graph.to_undirected()
        elif directed ==True:
            graph_mod = graph

        with tempfile.TemporaryDirectory(dir = './') as dname:
            original_graph = dname + '/node2vec_test.graph'
            emb_result = dname + '/node2vec_test.emb'
            graph_util.saveGraphToEdgeListTxtn2v(graph_mod, original_graph)
            args.append("-i:%s" % original_graph)
            args.append("-o:%s" % emb_result)
            args.append("-d:%d" % self._d)
            args.append("-l:%d" % self._walk_len)
            args.append("-r:%d" % self._num_walks)
            args.append("-k:%d" % self._con_size)
            args.append("-e:%d" % self._max_iter)
            args.append("-p:%f" % self._ret_p)
            args.append("-q:%f" % self._inout_p)
            args.append("-v")
            if directed ==True:
                args.append("-dr")
            if is_weighted ==True:
                args.append("-w")
            t1 = time()
            try:
                call(args)
            except Exception as e:
                print(str(e))
                raise Exception('node2vec not found. Please compile snap, place node2vec in the system path and grant executable permission')
            self._X = graph_util.loadEmbedding(emb_result)
            t2 = time()
        return self._X, (t2 - t1)
    
    def sample_random_walks(self, graph=None, edge_f=None,
                        is_weighted=False, no_python=False, directed=False):
        args = [HOME+"/snap/examples/node2vec/node2vec"]
        if directed == False:
            graph_mod = graph.to_undirected()
        elif directed ==True:
            graph_mod = graph

        with tempfile.TemporaryDirectory(dir = './') as dname:
            original_graph = dname + '/node2vec_test.graph'
            emb_result = dname + '/node2vec_test.walks'
            graph_util.saveGraphToEdgeListTxtn2v(graph_mod, original_graph)
            args.append("-i:%s" % original_graph)
            args.append("-o:%s" % emb_result)
            args.append("-d:%d" % self._d)
            args.append("-l:%d" % self._walk_len)
            args.append("-r:%d" % self._num_walks)
            args.append("-k:%d" % self._con_size)
            args.append("-e:%d" % self._max_iter)
            args.append("-p:%f" % self._ret_p)
            args.append("-q:%f" % self._inout_p)
            args.append("-v")
            if directed ==True:
                args.append("-dr")
            if is_weighted ==True:
                args.append("-w")
            args.append("-ow")
            t1 = time()
            try:
                call(args)
            except Exception as e:
                print(str(e))
                raise Exception('node2vec not found. Please compile snap, place node2vec in the system path and grant executable permission')
            self._W = loadWalks(emb_result)
            t2 = time()
        return self._W, (t2 - t1)
    
    def save_random_walks(self, graph=None, edge_f=None, is_weighted=False, no_python=False, directed=False,
                          save_directory=None, file_name=None, compress=False):
        args = [HOME+"/snap/examples/node2vec/node2vec"]
        if directed == False:
            graph_mod = graph.to_undirected()
        elif directed ==True:
            graph_mod = graph
            
        os.makedirs(save_directory, exist_ok=True)
        
        original_graph = save_directory + '/node2vec_test.graph'
        emb_result = save_directory + file_name
        graph_util.saveGraphToEdgeListTxtn2v(graph_mod, original_graph)
        args.append("-i:%s" % original_graph)
        args.append("-o:%s" % emb_result)
        args.append("-d:%d" % self._d)
        args.append("-l:%d" % self._walk_len)
        args.append("-r:%d" % self._num_walks)
        args.append("-k:%d" % self._con_size)
        args.append("-e:%d" % self._max_iter)
        args.append("-p:%f" % self._ret_p)
        args.append("-q:%f" % self._inout_p)
        args.append("-v")
        if directed ==True:
            args.append("-dr")
        if is_weighted ==True:
            args.append("-w")
        args.append("-ow")
        t1 = time()
        try:
            call(args)
        except Exception as e:
            print(str(e))
            raise Exception('node2vec not found. Please compile snap, place node2vec in the system path and grant executable permission')
        call('rm ' + original_graph, shell=True)
        
        if compress:
            f_in = open(emb_result)
            f_out = gzip.open(emb_result + '.gz', 'wt')
            f_out.writelines(f_in)
            f_out.close()
            f_in.close()
            call('rm ' + emb_result, shell=True)
        
        t2 = time()
        return (t2 - t1)

