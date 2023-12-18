import torch.nn.functional as F
import torch
from torch.nn import Linear, Dropout
import torch.nn as nn
import dgl
import json
from egat import EGATConv,EGATConvResid,EGATConvSA,EGATConvResidSA

class EGAT_Rxn(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # parse input parameters
        self.hidden_dim, self.num_heads = cfg.hidden_dim, cfg.num_heads
        self.aggregate = cfg.Aggregate
        self.getembeddings = cfg.Embed
        self.egatlayers = cfg.EGAT_layers

        self.selfattention = cfg.SA
        self.residualedition = cfg.Residual
        self.residbias = cfg.ResidBias

        num_node_feats = 17
        num_edge_feats = 15

        if cfg.removeelementinfo: num_node_feats = num_node_feats - 1

        if cfg.removeneighborcount and cfg.onlyH: num_node_feats = num_node_feats - 4
        elif cfg.removeneighborcount and not cfg.onlyH: num_node_feats = num_node_feats - 4
        elif not cfg.removeneighborcount and cfg.onlyH: num_node_feats = num_node_feats - 3
        
        if cfg.removereactiveinfo or cfg.molecular: num_node_feats = num_node_feats - 1

        if cfg.removeringinfo: 
            num_node_feats = num_node_feats - 1
            num_edge_feats = num_edge_feats - 1
        
        if cfg.removearomaticity: 
            num_node_feats = num_node_feats - 1
             
        if cfg.removeformalchargeinfo: num_node_feats = num_node_feats - 1
        if cfg.removechiralinfo: num_node_feats = num_node_feats - 3
        if cfg.removehybridinfo: num_node_feats = num_node_feats - 4
        if cfg.getradical: num_node_feats += 2
        if cfg.getspiro: num_node_feats += 1
        if cfg.getbridgehead: num_node_feats += 1
        if cfg.gethbinfo: num_node_feats += 2
        if cfg.geteneg: num_node_feats += 1
        
        if cfg.removebondorderinfo: num_edge_feats = num_edge_feats - 5
        if cfg.removebondtypeinfo: num_edge_feats = num_edge_feats - 5
        if cfg.removeconjinfo: num_edge_feats = num_edge_feats - 1
        if cfg.removestereoinfo: num_edge_feats = num_edge_feats - 3
        if cfg.getbondrot: num_edge_feats += 2
        self.getattentionmaps = cfg.AttentionMaps
        self.SA = cfg.SA
        # BLOCK1: massage-passing blocks
        if cfg.Resid is not None: 
            if not cfg.SA:
                self.egat1 = EGATConvResid(in_node_feats=num_node_feats,in_edge_feats=num_edge_feats,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads,edgeresid=cfg.Resid,bias=cfg.ResidBias)
                self.egat2 = EGATConvResid(in_node_feats=self.hidden_dim*self.num_heads,in_edge_feats=self.hidden_dim*self.num_heads,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads,edgeresid=cfg.Resid,bias=cfg.ResidBias)
            else:
                self.egat1 = EGATConvResidSA(in_node_feats=num_node_feats,in_edge_feats=num_edge_feats,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads,edgeresid=cfg.Resid,bias=cfg.ResidBias)
                self.egat2 = EGATConvResidSA(in_node_feats=self.hidden_dim*self.num_heads,in_edge_feats=self.hidden_dim*self.num_heads,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads,edgeresid=cfg.Resid,bias=cfg.ResidBias)
        else: 
            if not cfg.SA:   
                self.egat1 = EGATConv(in_node_feats=num_node_feats,in_edge_feats=num_edge_feats,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads)
                self.egat2 = EGATConv(in_node_feats=self.hidden_dim*self.num_heads,in_edge_feats=self.hidden_dim*self.num_heads,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads)
            else:
                self.egat1 = EGATConvSA(in_node_feats=num_node_feats,in_edge_feats=num_edge_feats,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads)
                self.egat2 = EGATConvSA(in_node_feats=self.hidden_dim*self.num_heads,in_edge_feats=self.hidden_dim*self.num_heads,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads)
        
        # BLOCK2: aggregate reactant and product nodes features
        self.agg_N_feats = nn.Sequential(nn.Linear(self.hidden_dim*self.num_heads, self.hidden_dim*self.num_heads, bias=True), nn.GELU())
        self.agg_E_feats = nn.Sequential(nn.Linear(self.hidden_dim*self.num_heads, self.hidden_dim*self.num_heads, bias=True), nn.GELU())

        # BLOCK3: final MLP layers
        self.mlp1 = nn.Sequential(nn.Linear(self.hidden_dim*self.num_heads*2, 256, bias=True),nn.GELU())
        self.mlp2 = nn.Sequential(nn.Linear(256, 128, bias=True),nn.GELU())
        self.mlp3 = nn.Linear(128, 1, bias=True)

    def forward(self, graphR,Radd):

        ##################################### 
        ############# layer one ############# 
        ##################################### 
        Rnode_feats, Redge_feats = self.egat1(graphR, graphR.ndata['x'], graphR.edata['x'])
        Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(),self.hidden_dim * self.num_heads)
        Redge_feats = Redge_feats.view(graphR.number_of_edges(),self.hidden_dim * self.num_heads)

        ##################################### 
        ############# layer N ############### 
        ##################################### 
        for i in range(self.egatlayers-1):
            Rnode_feats, Redge_feats = self.egat2(graphR, Rnode_feats, Redge_feats)
            Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(),self.hidden_dim * self.num_heads)
            Redge_feats = Redge_feats.view(graphR.number_of_edges(),self.hidden_dim * self.num_heads)

            if self.getattentionmaps and i == self.egatlayers:
                R_attn_scores = self.egat2.edge_attn
                if self.num_heads > 1: R_attn_scores = torch.norm(R_attn_scores,dim=1)
                graphR.edata['norm_attn'] = R_attn_scores
                if self.SA:
                    R_self_attn = self.egat2.self_attn
                    if self.num_heads > 1: R_self_attn = torch.norm(R_self_attn,dim=1)
                    graphR.ndata['norm_attn'] = R_self_attn
            
                # Initialize a square matrix with zeros
                matrix_size = graphR.number_of_nodes()
                R_combined_matrix = torch.zeros(matrix_size, matrix_size)
                # Fill the off-diagonal with edge attention scores
                src, dst = graphR.edges()
                R_combined_matrix[src, dst] = graphR.edata['norm_attn'].view(-1)

                if self.SA:
                    # Fill the diagonal with node self-attention scores
                    R_combined_matrix.fill_diagonal_(graphR.ndata['norm_attn'].view(-1))
        
        # merge R and P features
        Rxn_node_feature = self.agg_N_feats(Rnode_feats)
        Rxn_edge_feature = self.agg_E_feats(Redge_feats)

        # obtain global feature for layer 3
        graphR.ndata['x'] = Rxn_node_feature
        graphR.edata['x'] = Rxn_edge_feature 
        individual_graphs = dgl.unbatch(graphR)

        # Initialize a list to store the global features for each graph
        G_node_feats,G_edge_feats = [],[]

        # Iterate through the individual graphs
        for graph in individual_graphs:
            # Calculate the sum of the node features (assuming node features are stored in 'h')
            global_node_feature = graph.ndata['x'].sum(dim=0)
            global_edge_feature = graph.edata['x'].sum(dim=0)
            G_node_feats.append(global_node_feature)
            G_edge_feats.append(global_edge_feature)

        G_node_feats = torch.stack(G_node_feats)
        G_edge_feats = torch.stack(G_edge_feats)

        #################################### 
        ########## merge features ##########
        #################################### 
        G_features = torch.cat((G_node_feats,G_edge_feats), axis=1)
        self.G_features = G_features
        G_features = torch.cat((G_features,Radd), axis=1)
        # MLP
        x   = self.mlp1(G_features)
        x   = self.mlp2(x)
        x   = self.mlp3(x)

        if self.getembeddings == 0:
            if self.getattentionmaps:
                return x,R_combined_matrix
            else:
                return x
        elif self.getembeddings == 1:
            if self.getattentionmaps:
                return x,self.G_features,R_combined_matrix
            else:
                return x,self.G_features
        elif self.getembeddings == 2:
            if self.getattentionmaps:
                return self.G_features,R_combined_matrix
            else:
                return self.G_features
