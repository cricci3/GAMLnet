import torch

from torch.nn import Linear, ReLU, Softmax
from torch_geometric.nn import Sequential, GINConv, SAGEConv
from torchvision.ops import MLP
from models.GIN import GNN_GIN_Model
from models.SAGE import GNN_SAGE_Model

class GNN_GAMLNET_Model(torch.nn.Module):

    def __init__(self, hidden_size1: int,
                 hidden_size2: int,
                 input_size1: int, 
                 input_size2: int,
                 output_size: int,
                 num_layers1: int,
                 num_layers2: int,
                 gin_feature_indices):
        super().__init__()
        torch.manual_seed(12345)
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_layers1 = num_layers1
        self.num_layers2 = num_layers2
        self.input_size1 = input_size1
        self.input_size2 = input_size2 + hidden_size1 #here we do this because we concat output of gin and input of sage
        self.output_size = output_size
        self.gin_feature_indices = gin_feature_indices
        #self.mlp = Linear(in_size, output_size)
        #self.sm = Softmax(dim=0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        #NOTE: MAYBE I NEED TO REMOVE LINEAR LAYER AT THE END OF GIN
        self.gin_model =  GNN_GIN_Model(hidden_size=self.hidden_size1, input_size=self.input_size1, output_size=self.hidden_size1, num_layers=self.num_layers1)
        self.sage_model = GNN_SAGE_Model(hidden_size=self.hidden_size2, input_size=self.input_size2, output_size=self.output_size, num_layers=self.num_layers2)



    def forward(self, x, edge_index, edge_weight=None):
        # Message-passing: transform node features based on neighbors
        x1 = x
        if self.gin_feature_indices[0] == -1:
            x1 = torch.ones(x.size()[0], 1).to(self.device) #ONLY USE THIS IF NO FEATURES ARE WANTED
        elif self.gin_feature_indices == None:
            x1 = x
        else:
            # here we modify x to only include structural information
            x1 = x[:,self.gin_feature_indices]
        x1 = self.gin_model(x1, edge_index)

        #x1 = torch.nn.functional.normalize(x1,dim=0).to(torch.float32)
        x = torch.cat((x, x1), dim=1)
        
        out = self.sage_model(x, edge_index)

        return out

