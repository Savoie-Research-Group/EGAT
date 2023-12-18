import dgl;import torch as th;from edge_gat_resid import EGATConvResid

num_nodes, num_edges = 8,30;u, v = th.randint(num_nodes,size=(num_edges,)), th.randint(num_nodes,size=(num_edges,)) ;graph = dgl.graph((u,v))    

node_feats = th.rand((num_nodes, 20));edge_feats = th.rand((num_edges, 12))
egat = EGATConvResid(in_node_feats=20,in_edge_feats=12,out_node_feats=120,out_edge_feats=120,num_heads=4)
#forward pass                    
new_node_feats, new_edge_feats = egat(graph, node_feats, edge_feats)
