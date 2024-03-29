"""
Torch modules for graph Transformers with fully valuable edges (EGT). Modified to fit with DGL based on code from 
https://github.com/shamim-hussain/egt_pytorch/blob/master/lib/models/egt_layers.py. 
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional
import dgl.function as fn
from dgl.nn.functional import edge_softmax


class EGT(nn.Module):
    @staticmethod
    @torch.jit.script
    def _egt(scale_dot: bool,
             scale_degree: bool,
             num_heads: int,
             dot_dim: int,
             clip_logits_min: float,
             clip_logits_max: float,
             attn_dropout: float,
             attn_maskout: float,
             training: bool,
             num_vns: int,
             QKV: torch.Tensor,
             G: torch.Tensor,
             E: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        scale_dot (bool): Add a Scaling factor for the dot product for the Node Attention. 

        scale_degree (bool): Add a Scaling Factor to the dot product for the edge attention.
        num_heads (int): Number of heads to use
        dot_dim: (int): Dimension of the dot product.
        clip_logits_min: float,
        clip_logits_max: float,
        attn_dropout: float,
        attn_maskout: float,
        training: bool,
        num_vns: int,
        QKV: torch.Tensor,
        G: torch.Tensor,
        E: torch.Tensor

        
        """
        shp = QKV.shape
        # Get Q,K,V Vectors
        Q, K, V = QKV.view(shp[0],shp[1],-1,num_heads).split(dot_dim,dim=2)
        # Get Attention Vector
        A_hat = torch.einsum('bldh,bmdh->blmh', Q, K)
        # Scale the Attention by a scaling factor if needed
        if scale_dot:
            A_hat = A_hat * (dot_dim ** -0.5)
        # Add the edge term
        H_hat = A_hat.clamp(clip_logits_min, clip_logits_max) + E
        
        if attn_maskout > 0 and training:
            rmask = torch.empty_like(H_hat).bernoulli_(attn_maskout) * -1e9
            gates = torch.sigmoid(G)#+rmask
            A_tild = F.softmax(H_hat+rmask, dim=2) * gates
        else:
            gates = torch.sigmoid(G)
            A_tild = F.softmax(H_hat, dim=2) * gates
        
        
        if attn_dropout > 0:
            A_tild = F.dropout(A_tild, p=attn_dropout, training=training)
        
        V_att = torch.einsum('blmh,bmkh->blkh', A_tild, V)
        
        if scale_degree:
            degrees = torch.sum(gates,dim=2,keepdim=True)
            degree_scalers = torch.log(1+degrees)
            degree_scalers[:,:num_vns] = 1.
            V_att = V_att * degree_scalers
        
        V_att = V_att.reshape(shp[0],shp[1],num_heads*dot_dim)
        return V_att, H_hat

    @staticmethod
    @torch.jit.script
    def _egt_edge(scale_dot: bool,
                  num_heads: int,
                  dot_dim: int,
                  clip_logits_min: float,
                  clip_logits_max: float,
                  QK: torch.Tensor,
                  E: torch.Tensor) -> torch.Tensor:
        shp = QK.shape
        Q, K = QK.view(shp[0],shp[1],-1,num_heads).split(dot_dim,dim=2)
        
        A_hat = torch.einsum('bldh,bmdh->blmh', Q, K)
        if scale_dot:
            A_hat = A_hat * (dot_dim ** -0.5)
        H_hat = A_hat.clamp(clip_logits_min, clip_logits_max) + E
        return H_hat
    
    def __init__(self,
                 node_width                      ,
                 edge_width                      ,
                 num_heads                       ,
                 node_mha_dropout    = 0         ,
                 edge_mha_dropout    = 0         ,
                 node_ffn_dropout    = 0         ,
                 edge_ffn_dropout    = 0         ,
                 attn_dropout        = 0         ,
                 attn_maskout        = 0         ,
                 activation          = 'elu'     ,
                 clip_logits_value   = [-5,5]    ,
                 node_ffn_multiplier = 2.        ,
                 edge_ffn_multiplier = 2.        ,
                 scale_dot           = True      ,
                 scale_degree        = False     ,
                 node_update         = True      ,
                 edge_update         = True      ,
                 ):
        super().__init__()
        self.node_width          = node_width         
        self.edge_width          = edge_width          
        self.num_heads           = num_heads           
        self.node_mha_dropout    = node_mha_dropout        
        self.edge_mha_dropout    = edge_mha_dropout        
        self.node_ffn_dropout    = node_ffn_dropout        
        self.edge_ffn_dropout    = edge_ffn_dropout        
        self.attn_dropout        = attn_dropout
        self.attn_maskout        = attn_maskout
        self.activation          = activation          
        self.clip_logits_value   = clip_logits_value   
        self.node_ffn_multiplier = node_ffn_multiplier 
        self.edge_ffn_multiplier = edge_ffn_multiplier 
        self.scale_dot           = scale_dot
        self.scale_degree        = scale_degree        
        self.node_update         = node_update         
        self.edge_update         = edge_update        
        
        assert not (self.node_width % self.num_heads)
        self.dot_dim = self.node_width//self.num_heads
        
        self.mha_ln_h   = nn.LayerNorm(self.node_width)
        self.mha_ln_e   = nn.LayerNorm(self.edge_width)
        self.lin_E      = nn.Linear(self.edge_width, self.num_heads)
        if self.node_update:
            self.lin_QKV    = nn.Linear(self.node_width, self.node_width*3)
            self.lin_G      = nn.Linear(self.edge_width, self.num_heads)
        else:
            self.lin_QKV    = nn.Linear(self.node_width, self.node_width*2)
        
        self.ffn_fn     = getattr(F, self.activation)
        if self.node_update:
            self.lin_O_h    = nn.Linear(self.node_width, self.node_width)
            if self.node_mha_dropout > 0:
                self.mha_drp_h  = nn.Dropout(self.node_mha_dropout)
            
            node_inner_dim  = round(self.node_width*self.node_ffn_multiplier)
            self.ffn_ln_h   = nn.LayerNorm(self.node_width)
            self.lin_W_h_1  = nn.Linear(self.node_width, node_inner_dim)
            self.lin_W_h_2  = nn.Linear(node_inner_dim, self.node_width)
            if self.node_ffn_dropout > 0:
                self.ffn_drp_h  = nn.Dropout(self.node_ffn_dropout)
        
        if self.edge_update:
            self.lin_O_e    = nn.Linear(self.num_heads, self.edge_width)
            if self.edge_mha_dropout > 0:
                self.mha_drp_e  = nn.Dropout(self.edge_mha_dropout)
        
            edge_inner_dim  = round(self.edge_width*self.edge_ffn_multiplier)
            self.ffn_ln_e   = nn.LayerNorm(self.edge_width)
            self.lin_W_e_1  = nn.Linear(self.edge_width, edge_inner_dim)
            self.lin_W_e_2  = nn.Linear(edge_inner_dim, self.edge_width)
            if self.edge_ffn_dropout > 0:
                self.ffn_drp_e  = nn.Dropout(self.edge_ffn_dropout)
    
    def forward(self, g):
        g.edata['f'] = e
        g.ndata['h'] = h
        mask = g.mask
        
        h_r1 = h
        e_r1 = e
        
        h_ln = self.mha_ln_h(h)
        e_ln = self.mha_ln_e(e)
        
        QKV = self.lin_QKV(h_ln)
        E = self.lin_E(e_ln)
        
        if self.node_update:
            G = self.lin_G(e_ln)
            V_att, H_hat = self._egt(self.scale_dot,
                                     self.scale_degree,
                                     self.num_heads,
                                     self.dot_dim,
                                     self.clip_logits_value[0],
                                     self.clip_logits_value[1],
                                     self.attn_dropout,
                                     self.attn_maskout,
                                     self.training,
                                     0 if 'num_vns' not in g else g.num_vns,
                                     QKV,
                                     G, E)
            
            h = self.lin_O_h(V_att)
            if self.node_mha_dropout > 0:
                h = self.mha_drp_h(h)
            h += h_r1
            
            h_r2 = h
            h_ln = self.ffn_ln_h(h)
            h = self.lin_W_h_2(self.ffn_fn(self.lin_W_h_1(h_ln)))
            if self.node_ffn_dropout > 0:
                h = self.ffn_drp_h(h)
            h += h_r2
        else:
            H_hat = self._egt_edge(self.scale_dot,
                                   self.num_heads,
                                   self.dot_dim,
                                   self.clip_logits_value[0],
                                   self.clip_logits_value[1],
                                   QKV, E)
        
        
        if self.edge_update:
            e = self.lin_O_e(H_hat)
            if self.edge_mha_dropout > 0:
                e = self.mha_drp_e(e)
            e += e_r1
            
            e_r2 = e
            e_ln = self.ffn_ln_e(e)
            e = self.lin_W_e_2(self.ffn_fn(self.lin_W_e_1(e_ln)))
            if self.edge_ffn_dropout > 0:
                e = self.ffn_drp_e(e)
            e += e_r2
        
        g = g.copy()
        g.edata['f'] = e
        g.ndata['h'] = h
        return g.ndata.pop('h'), g.edata.pop('f')
    
    


