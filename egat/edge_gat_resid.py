"""Torch modules for graph attention networks with fully valuable edges (EGAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
from copy import deepcopy
import torch as th
from torch import nn
from torch.nn import init

# pylint: enable=W0235
class EGATConvResid(nn.Module):
    r"""
    
    Description
    -----------
    Apply Graph Attention Layer over input graph. EGAT is an extension
    of regular `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__ 
    handling edge features, detailed description is available in
    `Rossmann-Toolbox <https://pubmed.ncbi.nlm.nih.gov/34571541/>`__ (see supplementary data).
     The difference appears in the method how unnormalized attention scores :math:`e_{ij}`
     are obtain:
        
    .. math::
        e_{ij} &= \vec{F} (f_{ij}^{\prime})

        f_{ij}^{\prim} &= \mathrm{LeakyReLU}\left(A [ h_{i} \| f_{ij} \| h_{j}]\right)

    where :math:`f_{ij}^{\prim}` are edge features, :math:`\mathrm{A}` is weight matrix and 
    :math: `\vec{F}` is weight vector. After that resulting node features 
    :math:`h_{i}^{\prim}` are updated in the same way as in regular GAT. 
   
    Parameters
    ----------
    in_node_feats : int
        Input node feature size :math:`h_{i}`.
    in_edge_feats : int
        Input edge feature size :math:`f_{ij}`.
    out_node_feats : int
        Output nodes feature size.
    out_edge_feats : int
        Output edge feature size.
    num_heads : int
        Number of attention heads.
        
    Examples
    ----------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import EGATConv
    >>> 
    >>> num_nodes, num_edges = 8, 30
    >>>#define connections
    >>> u, v = th.randint(num_nodes, num_edges), th.randint(num_nodes, num_edges) 
    >>> graph = dgl.graph((u,v))    

    >>> node_feats = th.rand((num_nodes, 20)) 
    >>> edge_feats = th.rand((num_edges, 12))
    >>> egat = EGATConv(in_node_feats=20,
                          in_edge_feats=12,
                          out_node_feats=15,
                          out_edge_feats=10,
                          num_heads=3)
    >>> #forward pass                    
    >>> new_node_feats, new_edge_feats = egat(graph, node_feats, edge_feats)
    >>> new_node_feats.shape, new_edge_feats.shape
    ((8, 3, 12), (30, 3, 10))
    """
    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 out_edge_feats,
                 num_heads,edgeresid=None,bias=False,**kw_args):
        
        super().__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self._in_node_feats = in_node_feats
        self._in_edge_feats = in_edge_feats
        
        self._edge_resid_setting = edgeresid
        self.fc_nodes = nn.Linear(in_node_feats, out_node_feats*num_heads, bias=True)
        self.fc_edges = nn.Linear(in_edge_feats + 2*in_node_feats, out_edge_feats*num_heads, bias=True)
        self.fc_attn = nn.Linear(out_edge_feats, num_heads, bias=False)
        if self._edge_resid_setting == 'v1':
            self.fc_edgeresid = nn.Linear(in_edge_feats, out_edge_feats*num_heads, bias=bias)
        elif self._edge_resid_setting == 'gate':
            self.fc_nodes_gate1 = nn.Linear(in_node_feats, out_node_feats*num_heads, bias=bias)
            self.fc_nodes_gate2 = nn.Linear(in_node_feats, out_node_feats*num_heads, bias=bias)
            self.fc_edges_gate1 = nn.Linear(in_node_feats, out_node_feats*num_heads, bias=bias)
            self.fc_edges_gate2 = nn.Linear(in_node_feats, out_node_feats*num_heads, bias=bias)
            

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fc_nodes.weight, gain=gain)
        init.xavier_normal_(self.fc_edges.weight, gain=gain)
        init.xavier_normal_(self.fc_attn.weight, gain=gain)
        if self._edge_resid_setting == 'v1':
            init.xavier_normal_(self.fc_edgeresid.weight, gain=gain)
        elif self._edge_resid_setting == 'gate':
            init.xavier_normal_(self.fc_nodes_gate1.weight, gain=gain)
            init.xavier_normal_(self.fc_nodes_gate2.weight, gain=gain)
            init.xavier_normal_(self.fc_edges_gate1.weight, gain=gain)
            init.xavier_normal_(self.fc_edges_gate2.weight, gain=gain)
            

    def edge_attention(self, edges):
        #extract features
        h_src = edges.src['h']
        h_dst = edges.dst['h']
        f = edges.data['f']
        #stack h_i | f_ij | h_j
        stack = th.cat([h_src, f, h_dst], dim=-1)
        # apply FC and activation
        f_out = self.fc_edges(stack)
        if self._out_node_feats != self._in_node_feats:
            if self._edge_resid_setting == 'v1':
                f_resid = self.fc_edgeresid(f)
                f_resid = f_resid.view(-1, self._num_heads, self._out_edge_feats)
            elif self._edge_resid_setting == 'v2':
                f_resid = deepcopy(f_out)
                f_resid = f_resid.view(-1, self._num_heads, self._out_edge_feats)
        else:
            if self._edge_resid_setting == 'v1' or self._edge_resid_setting == 'v2':
                f_resid = deepcopy(f)
                f_resid = f_resid.view(-1, self._num_heads, self._out_edge_feats)
        f_out = nn.functional.leaky_relu(f_out)
        f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)
        # apply FC to reduce edge_feats to scalar
        a = self.fc_attn(f_out).sum(-1).unsqueeze(-1)
        self.edge_attn = a
        if self._edge_resid_setting in ['v1','v2','skip','gate']:
            return {'a': a, 'f' : f_out,'f_resid': f_resid}
        else:
            return {'a': a, 'f' : f_out}


    def message_func(self, edges):
        return {'h': edges.src['h'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        alpha = nn.functional.softmax(nodes.mailbox['a'], dim=1)
        h = th.sum(alpha * nodes.mailbox['h'], dim=1)
        return {'h': h}

    def node_gate_connector(self,newnodes,nodes):

        if len(newnodes.shape) == 3:
            shapevector = deepcopy(newnodes.shape)
            newnodes = newnodes.view([shapevector[0],shapevector[1]*shapevector[2]])

        if len(nodes.shape) == 3:
            shapevector = deepcopy(nodes.shape)
            nodes = nodes.view([shapevector[0],shapevector[1]*shapevector[2]])

        num_atoms = nodes.shape[0]
        dim = nodes.shape[1]
        _b = nn.Parameter(nn.init.xavier_uniform_(th.empty(dim, dtype=th.float64)))

        # Tiling and reshaping the parameter to create a 2D tensor
        _b = _b.unsqueeze(0).expand(num_atoms, -1)  # Tile across rows
        _b = _b.view(num_atoms, dim)  # Reshape to [num_atoms, dim]

        newnodes = self.fc_nodes_gate1(newnodes)
        nodes = self.fc_nodes_gate2(nodes)

        gate = th.sigmoid(newnodes + nodes + _b)

        newnodes = newnodes * gate + nodes * (1.0 - gate)
        
        if len(newnodes.shape) == 2:
            newnodes = newnodes.view([newnodes.shape[0],self._num_heads,self._out_node_feats])

        return newnodes
    
    def edge_gate_connector(self,newedges,edges):

        if len(newedges.shape) == 3:
            shapevector = deepcopy(newedges.shape)
            newedges = newedges.view([shapevector[0],shapevector[1]*shapevector[2]])

        if len(edges.shape) == 3:
            shapevector = deepcopy(edges.shape)
            edges = edges.view([shapevector[0],shapevector[1]*shapevector[2]])

        num_atoms = edges.shape[0]
        dim = edges.shape[1]
        _b = nn.Parameter(nn.init.xavier_uniform_(th.empty(dim, dtype=th.float64)))

        # Tiling and reshaping the parameter to create a 2D tensor
        _b = _b.unsqueeze(0).expand(num_atoms, -1)  # Tile across rows
        _b = _b.view(num_atoms, dim)  # Reshape to [num_atoms, dim]

        newedges = self.fc_nodes_gate1(newedges)
        edges = self.fc_nodes_gate2(edges)

        gate = th.sigmoid(newedges + edges + _b)

        newedges = newedges * gate + edges * (1.0 - gate)
        
        if len(newedges.shape) == 2:
            newedges = newedges.view([newedges.shape[0],self._num_heads,self._out_edge_feats])

        return newedges



    def forward(self, graph, nfeats, efeats):
        r"""
        Compute new node and edge features.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        nfeats : torch.Tensor
            The input node feature of shape :math:`(*, D_{in})`
            where:
                :math:`D_{in}` is size of input node feature,
                :math:`*` is the number of nodes.
        efeats: torch.Tensor
             The input edge feature of shape :math:`(*, F_{in})`
             where:
                 :math:`F_{in}` is size of input node feauture,
                 :math:`*` is the number of edges.
       
            
        Returns
        -------
        pair of torch.Tensor
            node output features followed by edge output features
            The node output feature of shape :math:`(*, H, D_{out})` 
            The edge output feature of shape :math:`(*, H, F_{out})`
            where:
                :math:`H` is the number of heads,
                :math:`D_{out}` is size of output node feature,
                :math:`F_{out}` is size of output edge feature.            
        """
        
        with graph.local_scope():
        ##TODO allow node src and dst feats
            graph.edata['f'] = efeats
            graph.ndata['h'] = nfeats

            graph.apply_edges(self.edge_attention)

            nfeats_ = self.fc_nodes(nfeats)
            nfeats_ = nfeats_.view(-1, self._num_heads, self._out_node_feats)
            
            graph.ndata['h'] = nfeats_
            

            graph.update_all(message_func = self.message_func,
                         reduce_func = self.reduce_func)
            
            if self._out_node_feats != self._in_node_feats:
                if self._edge_resid_setting == 'v1' or self._edge_resid_setting == 'v2':
                    graph.ndata['h'] += nfeats_
                    graph.edata['f'] += graph.edata['f_resid']
                elif self._edge_resid_setting == 'skip':
                    graph.ndata['h'] += nfeats_
                    graph.ndata['h'] = nn.functional.relu(graph.ndata['h'])

                    graph.edata['f'] += graph.edata['f_resid']
                    graph.edata['f'] = nn.functional.relu(graph.edata['f'])
                elif self._edge_resid_setting == 'gate':    
                    graph.ndata['h'] = self.node_gate_connector(graph.ndata['h'],nfeats_)
                    graph.ndata['f'] = self.edge_gate_connector(graph.ndata['f'],graph.edata['f_resid'])

            else:
                if self._edge_resid_setting == 'v1' or self._edge_resid_setting == 'v2':
                    graph.ndata['h'] += nfeats
                    graph.edata['f'] += graph.edata['f_resid']
                elif self._edge_resid_setting == 'skip':
                    graph.ndata['h'] += nfeats
                    graph.ndata['h'] = nn.functional.relu(graph.ndata['h'])

                    graph.edata['f'] += graph.edata['f_resid']
                    graph.edata['f'] = nn.functional.relu(graph.edata['f'])
                elif self._edge_resid_setting == 'gate':
                    graph.ndata['h'] = self.node_gate_connector(graph.ndata['h'],nfeats)
                    graph.ndata['f'] = self.edge_gate_connector(graph.ndata['f'],graph.edata['f_resid'])

            return graph.ndata.pop('h'), graph.edata.pop('f')
    
