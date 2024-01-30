import torch.nn.functional as F
import torch
from torch.nn import Linear, Dropout
import torch.nn as nn
import dgl
import json

class MolecularAblationModel(nn.Module):
    def __init__(self, cfg, egat_model, nn_model):
        super(MolecularAblationModel, self).__init__()

        # parse input parameters
        self.hidden_dim, self.num_heads = cfg.hidden_dim, cfg.num_heads
        self.aggregate = cfg.Aggregate
        self.getembeddings = cfg.Embed
        self.egatlayers = cfg.EGAT_layers

        self.selfattention = cfg.SA
        self.residualedition = cfg.Resid
        self.residbias = cfg.ResidBias

        if not cfg.useFullHyb: num_node_feats = 17
        else: num_node_feats = 22
        if not cfg.useOld: num_edge_feats = 15
        else: num_edge_feats = 14

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
        if cfg.removebondtypeinfo or cfg.molecular: 
            if cfg.useOld:
                num_edge_feats = num_edge_feats - 4
            else:
                num_edge_feats = num_edge_feats - 5
        if cfg.removeconjinfo: num_edge_feats = num_edge_feats - 1
        if cfg.removestereoinfo: num_edge_feats = num_edge_feats - 3
        if cfg.getbondrot: num_edge_feats += 2


        # Copy EGAT layers from egat_model
        self.egat1 = egat_model.egat1
        self.egat2 = egat_model.egat2

        # Copy NN layers from nn_model
        self.agg_N_feats = egat_model.agg_N_feats
        self.agg_E_feats = egat_model.agg_E_feats
        self.mlp1 = nn_model.mlp1
        self.mlp2 = nn_model.mlp2
        self.mlp3 = nn_model.mlp3

    def forward(self, graphR):
        # Forward pass through EGAT layers
        Rnode_feats, Redge_feats = self.egat1(graphR, graphR.ndata['x'], graphR.edata['x'])
        Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(), self.hidden_dim * self.num_heads)
        Redge_feats = Redge_feats.view(graphR.number_of_edges(), self.hidden_dim * self.num_heads)

        for i in range(self.egatlayers-1):
            Rnode_feats, Redge_feats = self.egat2(graphR, Rnode_feats, Redge_feats)
            Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(), self.hidden_dim * self.num_heads)
            Redge_feats = Redge_feats.view(graphR.number_of_edges(), self.hidden_dim * self.num_heads)


            if self.getattentionmaps and i == self.egatlayers:
                R_attn_scores = self.egat2.edge_attn
                if self.num_heads > 1:R_attn_scores = torch.norm(R_attn_scores,dim=1)
                graphR.edata['norm_attn'] = R_attn_scores
                if self.SA:
                    R_self_attn = self.egat2.self_attn
                    if self.num_heads > 1:R_self_attn = torch.norm(R_self_attn,dim=1)
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

        # Aggregate R and P features
        Rxn_node_feature = self.agg_N_feats(Rnode_feats)
        Rxn_edge_feature = self.agg_E_feats(Redge_feats)

        # Forward pass through NN layers
        graphR.ndata['x'] = Rxn_node_feature
        graphR.edata['x'] = Rxn_edge_feature
        individual_graphs = dgl.unbatch(graphR)

        G_node_feats, G_edge_feats = [], []

        for graph in individual_graphs:
            global_node_feature = graph.ndata['x'].sum(dim=0)
            global_edge_feature = graph.edata['x'].sum(dim=0)
            G_node_feats.append(global_node_feature)
            G_edge_feats.append(global_edge_feature)

            

        G_node_feats = torch.stack(G_node_feats)
        G_edge_feats = torch.stack(G_edge_feats)

        # Merge features
        G_features = torch.cat((G_node_feats, G_edge_feats), axis=1)
        # MLP
        x = self.mlp1(G_features)
        x = self.mlp2(x)
        x = self.mlp3(x)

        if self.getembeddings == 0:
            if self.getattentionmaps:
                return x,R_combined_matrix
            else:
                return x
        elif self.getembeddings == 1:
            if self.getattentionmaps:
                return x,G_features,R_combined_matrix
            else:
                return x,G_features
        elif self.getembeddings == 2:
            if self.getattentionmaps:
                return G_features,R_combined_matrix
            else:
                return G_features

class ReactionAblationModel(nn.Module):
    def __init__(self, cfg, egat_model, nn_model):
        super(ReactionAblationModel, self).__init__()

        # parse input parameters
        self.hidden_dim, self.num_heads = cfg.hidden_dim, cfg.num_heads
        self.aggregate = cfg.Aggregate
        self.getembeddings = cfg.Embed
        self.egatlayers = cfg.EGAT_layers

        self.selfattention = cfg.SA
        self.residualedition = cfg.Resid
        self.residbias = cfg.ResidBias

        if not cfg.useFullHyb: num_node_feats = 17
        else: num_node_feats = 22
        if not cfg.useOld: num_edge_feats = 15
        else: num_edge_feats = 14

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
        if cfg.removebondtypeinfo or cfg.molecular: 
            if cfg.useOld:
                num_edge_feats = num_edge_feats - 4
            else:
                num_edge_feats = num_edge_feats - 5
        if cfg.removeconjinfo: num_edge_feats = num_edge_feats - 1
        if cfg.removestereoinfo: num_edge_feats = num_edge_feats - 3
        if cfg.getbondrot: num_edge_feats += 2


        # Copy EGAT layers from egat_model
        self.egat1 = egat_model.egat1
        self.egat2 = egat_model.egat2

        # Copy NN layers from nn_model
        self.agg_N_feats = egat_model.agg_N_feats
        self.agg_E_feats = egat_model.agg_E_feats
        self.mlp1 = nn_model.mlp1
        self.mlp2 = nn_model.mlp2
        self.mlp3 = nn_model.mlp3

    def forward(self, graphR,graphP):
        # Forward pass through EGAT layers
        Rnode_feats, Redge_feats = self.egat1(graphR, graphR.ndata['x'], graphR.edata['x'])
        Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(), self.hidden_dim * self.num_heads)
        Redge_feats = Redge_feats.view(graphR.number_of_edges(), self.hidden_dim * self.num_heads)

        Pnode_feats, Pedge_feats = self.egat1(graphP, graphP.ndata['x'], graphP.edata['x'])
        Pnode_feats = Pnode_feats.view(graphP.number_of_nodes(), self.hidden_dim * self.num_heads)
        Pedge_feats = Pedge_feats.view(graphP.number_of_edges(), self.hidden_dim * self.num_heads)

        for i in range(self.egatlayers-1):
            Rnode_feats, Redge_feats = self.egat2(graphR, Rnode_feats, Redge_feats)
            Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(), self.hidden_dim * self.num_heads)
            Redge_feats = Redge_feats.view(graphR.number_of_edges(), self.hidden_dim * self.num_heads)

            if self.getattentionmaps and i == self.egatlayers:
                R_attn_scores = self.egat2.edge_attn
                if self.num_heads > 1:R_attn_scores = torch.norm(R_attn_scores,dim=1)
                graphR.edata['norm_attn'] = R_attn_scores
                if self.SA:
                    R_self_attn = self.egat2.self_attn
                    if self.num_heads > 1:R_self_attn = torch.norm(R_self_attn,dim=1)
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

            Pnode_feats, Pedge_feats = self.egat1(graphP, graphP.ndata['x'], graphP.edata['x'])
            Pnode_feats = Pnode_feats.view(graphP.number_of_nodes(), self.hidden_dim * self.num_heads)
            Pedge_feats = Pedge_feats.view(graphP.number_of_edges(), self.hidden_dim * self.num_heads)

            if self.getattentionmaps and i == self.egatlayers:
                P_attn_scores = self.egat2.edge_attn
                if self.num_heads > 1:P_attn_scores = torch.norm(P_attn_scores,dim=1)
                graphP.edata['norm_attn'] = P_attn_scores
                if self.SA:
                    P_self_attn = self.egat2.self_attn
                    if self.num_heads > 1: P_self_attn = torch.norm(P_self_attn,dim=1)
                    graphR.ndata['norm_attn'] = P_self_attn
                
                # Initialize a square matrix with zeros
                matrix_size = graphP.number_of_nodes()
                P_combined_matrix = torch.zeros(matrix_size, matrix_size)
                # Fill the off-diagonal with edge attention scores
                src, dst = graphP.edges()
                P_combined_matrix[src, dst] = graphP.edata['norm_attn'].view(-1)

                if self.SA:
                    # Fill the diagonal with node self-attention scores
                    P_combined_matrix.fill_diagonal_(graphP.ndata['norm_attn'].view(-1))

        # Aggregate R and P features
        if self.aggregate == 'Concat':
            Rxn_node_feature = self.agg_N_feats(torch.cat(Pnode_feats,Rnode_feats))
            Rxn_edge_feature = self.agg_E_feats(torch.cat(Pedge_feats,Redge_feats))
        else:
            Rxn_node_feature = self.agg_N_feats(Pnode_feats - Rnode_feats)
            Rxn_edge_feature = self.agg_E_feats(Pedge_feats - Redge_feats)

        # Forward pass through NN layers
        graphR.ndata['x'] = Rxn_node_feature
        graphR.edata['x'] = Rxn_edge_feature
        individual_graphs = dgl.unbatch(graphR)

        G_node_feats, G_edge_feats = [], []

        for graph in individual_graphs:
            global_node_feature = graph.ndata['x'].sum(dim=0)
            global_edge_feature = graph.edata['x'].sum(dim=0)
            G_node_feats.append(global_node_feature)
            G_edge_feats.append(global_edge_feature)

        G_node_feats = torch.stack(G_node_feats)
        G_edge_feats = torch.stack(G_edge_feats)

        # Merge features
        G_features = torch.cat((G_node_feats, G_edge_feats), axis=1)

        # MLP
        x = self.mlp1(G_features)
        x = self.mlp2(x)
        x = self.mlp3(x)

        if self.getembeddings == 0:
            if self.getattentionmaps:
                return x,R_combined_matrix,P_combined_matrix
            else:
                return x
        elif self.getembeddings == 1:
            if self.getattentionmaps:
                return x,G_features,R_combined_matrix,P_combined_matrix
            else:
                return x,G_features
        elif self.getembeddings == 2:
            if self.getattentionmaps:
                return G_features,R_combined_matrix,P_combined_matrix
            else:
                return G_features


class MolecularAblationModelwAddOns(nn.Module):
    def __init__(self, cfg, egat_model, nn_model):
        super(MolecularAblationModel, self).__init__()

        # parse input parameters
        self.hidden_dim, self.num_heads = cfg.hidden_dim, cfg.num_heads
        self.aggregate = cfg.Aggregate
        self.getembeddings = cfg.Embed
        self.egatlayers = cfg.EGAT_layers

        self.selfattention = cfg.SA
        self.residualedition = cfg.Resid
        self.residbias = cfg.ResidBias

        if not cfg.useFullHyb: num_node_feats = 17
        else: num_node_feats = 22
        if not cfg.useOld: num_edge_feats = 15
        else: num_edge_feats = 14

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
        if cfg.removebondtypeinfo or cfg.molecular: 
            if cfg.useOld:
                num_edge_feats = num_edge_feats - 4
            else:
                num_edge_feats = num_edge_feats - 5
        if cfg.removeconjinfo: num_edge_feats = num_edge_feats - 1
        if cfg.removestereoinfo: num_edge_feats = num_edge_feats - 3
        if cfg.getbondrot: num_edge_feats += 2


        # Copy EGAT layers from egat_model
        self.egat1 = egat_model.egat1
        self.egat2 = egat_model.egat2

        # Copy NN layers from nn_model
        self.agg_N_feats = egat_model.agg_N_feats
        self.agg_E_feats = egat_model.agg_E_feats
        self.mlp1 = nn_model.mlp1
        self.mlp2 = nn_model.mlp2
        self.mlp3 = nn_model.mlp3

    def forward(self, graphR,Radd):
        # Forward pass through EGAT layers
        Rnode_feats, Redge_feats = self.egat1(graphR, graphR.ndata['x'], graphR.edata['x'])
        Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(), self.hidden_dim * self.num_heads)
        Redge_feats = Redge_feats.view(graphR.number_of_edges(), self.hidden_dim * self.num_heads)

        for i in range(self.egatlayers-1):
            Rnode_feats, Redge_feats = self.egat2(graphR, Rnode_feats, Redge_feats)
            Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(), self.hidden_dim * self.num_heads)
            Redge_feats = Redge_feats.view(graphR.number_of_edges(), self.hidden_dim * self.num_heads)


            if self.getattentionmaps and i == self.egatlayers:
                R_attn_scores = self.egat2.edge_attn
                if self.num_heads > 1:R_attn_scores = torch.norm(R_attn_scores,dim=1)
                graphR.edata['norm_attn'] = R_attn_scores
                if self.SA:
                    R_self_attn = self.egat2.self_attn
                    if self.num_heads > 1:R_self_attn = torch.norm(R_self_attn,dim=1)
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

        # Aggregate R and P features
        Rxn_node_feature = self.agg_N_feats(Rnode_feats)
        Rxn_edge_feature = self.agg_E_feats(Redge_feats)

        # Forward pass through NN layers
        graphR.ndata['x'] = Rxn_node_feature
        graphR.edata['x'] = Rxn_edge_feature
        individual_graphs = dgl.unbatch(graphR)

        G_node_feats, G_edge_feats = [], []

        for graph in individual_graphs:
            global_node_feature = graph.ndata['x'].sum(dim=0)
            global_edge_feature = graph.edata['x'].sum(dim=0)
            G_node_feats.append(global_node_feature)
            G_edge_feats.append(global_edge_feature)

            

        G_node_feats = torch.stack(G_node_feats)
        G_edge_feats = torch.stack(G_edge_feats)

        # Merge features
        G_features = torch.cat((G_node_feats, G_edge_feats), axis=1)
        self.G_features = G_features
        G_features = torch.cat((G_features,Radd), axis=1)
        # MLP
        x = self.mlp1(G_features)
        x = self.mlp2(x)
        x = self.mlp3(x)

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


class ReactionAblationModelwAddOns(nn.Module):
    def __init__(self, cfg, egat_model, nn_model):
        super(ReactionAblationModel, self).__init__()

        # parse input parameters
        self.hidden_dim, self.num_heads = cfg.hidden_dim, cfg.num_heads
        self.aggregate = cfg.Aggregate
        self.getembeddings = cfg.Embed
        self.egatlayers = cfg.EGAT_layers
        self.aggregaterdkit = cfg.AddOnAgg

        self.selfattention = cfg.SA
        self.residualedition = cfg.Resid
        self.residbias = cfg.ResidBias

        if not cfg.useFullHyb: num_node_feats = 17
        else: num_node_feats = 22
        if not cfg.useOld: num_edge_feats = 15
        else: num_edge_feats = 14

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
        if cfg.removebondtypeinfo or cfg.molecular: 
            if cfg.useOld:
                num_edge_feats = num_edge_feats - 4
            else:
                num_edge_feats = num_edge_feats - 5
        if cfg.removeconjinfo: num_edge_feats = num_edge_feats - 1
        if cfg.removestereoinfo: num_edge_feats = num_edge_feats - 3
        if cfg.getbondrot: num_edge_feats += 2


        # Copy EGAT layers from egat_model
        self.egat1 = egat_model.egat1
        self.egat2 = egat_model.egat2

        # Copy NN layers from nn_model
        self.agg_N_feats = egat_model.agg_N_feats
        self.agg_E_feats = egat_model.agg_E_feats
        self.mlp1 = nn_model.mlp1
        self.mlp2 = nn_model.mlp2
        self.mlp3 = nn_model.mlp3

    def forward(self, graphR, graphP,Radd,Padd):

        ##################################### 
        ############# layer one ############# 
        ##################################### 
        Rnode_feats, Redge_feats = self.egat1(graphR, graphR.ndata['x'], graphR.edata['x'])
        Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(),self.hidden_dim * self.num_heads)
        Redge_feats = Redge_feats.view(graphR.number_of_edges(),self.hidden_dim * self.num_heads)

        Pnode_feats, Pedge_feats = self.egat1(graphP, graphP.ndata['x'], graphP.edata['x'])
        Pnode_feats = Pnode_feats.view(graphP.number_of_nodes(),self.hidden_dim * self.num_heads)
        Pedge_feats = Pedge_feats.view(graphP.number_of_edges(),self.hidden_dim * self.num_heads)

        ##################################### 
        ############# layer N ############### 
        ##################################### 
        for i in range(self.egatlayers-1):
            Rnode_feats, Redge_feats = self.egat2(graphR, Rnode_feats, Redge_feats)
            Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(),self.hidden_dim * self.num_heads)
            Redge_feats = Redge_feats.view(graphR.number_of_edges(),self.hidden_dim * self.num_heads)
            
            if self.getattentionmaps and i == self.egatlayers:
                R_attn_scores = self.egat2.edge_attn
                if self.num_heads > 1:R_attn_scores = torch.norm(R_attn_scores,dim=1)
                graphR.edata['norm_attn'] = R_attn_scores
                if self.SA:
                    R_self_attn = self.egat2.self_attn
                    if self.num_heads > 1:R_self_attn = torch.norm(R_self_attn,dim=1)
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

        
            Pnode_feats, Pedge_feats = self.egat2(graphP, Pnode_feats, Pedge_feats)
            Pnode_feats = Pnode_feats.view(graphP.number_of_nodes(),self.hidden_dim * self.num_heads)
            Pedge_feats = Pedge_feats.view(graphP.number_of_edges(),self.hidden_dim * self.num_heads)

            if self.getattentionmaps and i == self.egatlayers:
                P_attn_scores = self.egat2.edge_attn
                if self.num_heads > 1:P_attn_scores = torch.norm(P_attn_scores,dim=1)
                graphP.edata['norm_attn'] = P_attn_scores
                if self.SA:
                    P_self_attn = self.egat2.self_attn
                    if self.num_heads > 1: P_self_attn = torch.norm(P_self_attn,dim=1)
                    graphR.ndata['norm_attn'] = P_self_attn
                
                # Initialize a square matrix with zeros
                matrix_size = graphP.number_of_nodes()
                P_combined_matrix = torch.zeros(matrix_size, matrix_size)
                # Fill the off-diagonal with edge attention scores
                src, dst = graphP.edges()
                P_combined_matrix[src, dst] = graphP.edata['norm_attn'].view(-1)

                if self.SA:
                    # Fill the diagonal with node self-attention scores
                    P_combined_matrix.fill_diagonal_(graphP.ndata['norm_attn'].view(-1))
            

            

        

        # merge R and P features
        if self.aggregate == 'Concat':
            Rxn_node_feature = self.agg_N_feats(torch.cat(Pnode_feats,Rnode_feats))
            Rxn_edge_feature = self.agg_E_feats(torch.cat(Pedge_feats,Redge_feats))
        else:
            Rxn_node_feature = self.agg_N_feats(Pnode_feats - Rnode_feats)
            Rxn_edge_feature = self.agg_E_feats(Pedge_feats - Redge_feats)

        # obtain global feature for larer 3
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

        if self.aggregaterdkit == False:
            G_features = torch.cat((G_features,Padd-Radd), axis=1)
        else:
            G_features = torch.cat((G_features,Padd,Radd), axis=1)
        # MLP
        x   = self.mlp1(G_features)
        x   = self.mlp2(x)
        x   = self.mlp3(x)

        if self.getembeddings == 0:
            if self.getattentionmaps:
                return x,R_combined_matrix,P_combined_matrix
            else:
                return x
        elif self.getembeddings == 1:
            if self.getattentionmaps:
                return x,self.G_features,R_combined_matrix,P_combined_matrix
            else:
                return x,G_features
        elif self.getembeddings == 2:
            if self.getattentionmaps:
                return self.G_features,R_combined_matrix,P_combined_matrix
            else:
                return self.G_features




'''
# Usage
egat_model_instance = EGAT_Rxn(cfg_egat)  # Replace with your actual config
nn_model_instance = EGAT_Rxn(cfg_nn)      # Replace with your actual config

# Create the combined model
combined_model = CustomCombinedModel(egat_model_instance, nn_model_instance)

# Forward pass with graphR (assuming it's defined)
output = combined_model(graphR)
''' 
