import torch

from torch.nn import Linear, ReLU, Softmax
from torch_geometric.nn import Sequential, GINConv, SAGEConv
from torchvision.ops import MLP
from GIN3 import GNN_GIN_Model
from SAGE import GNN_SAGE_Model

class GNN_GINSAGE_Model(torch.nn.Module):

    def __init__(self, hidden_size: int,
                 input_size: int, 
                 output_size: int,
                 num_layers1: int,
                 num_layers2: int,
                 data_GIN_mask):
        super().__init__()
        torch.manual_seed(12345)
        self.hidden_size = hidden_size
        self.num_layers1 = num_layers1
        self.num_layers2 = num_layers2
        self.input_size = input_size
        self.output_size = output_size
        self.data_GIN_mask = data_GIN_mask
        #self.mlp = Linear(in_size, output_size)
        #self.sm = Softmax(dim=0)

        self.convs1 = torch.nn.ModuleList()
        self.convs2 = torch.nn.ModuleList()
        for l in range(num_layers1):
            #in_size = dataset_num_features if l == 0 else hidden_size
            in_size = 1 if l == 0 else hidden_size
            
            conv = GINConv(in_channels=in_size, out_channels=hidden_size,nn= Linear(in_size, hidden_size))
            #conv = GINConv(in_channels=in_size, out_channels=hidden_size,nn= MLP(in_size, [hidden_size,hidden_size]))
            
            #mpnn = MPNN(in_channels=in_size, out_channels=hidden_size, in_channels2=in_size2)
            self.convs1.append(conv)
        for l in range(num_layers2):
            #in_size = dataset_num_features if l == 0 else hidden_size
            #in_size = input_size if l == 0 else hidden_size
            in_size = hidden_size + input_size if l == 0 else hidden_size
            conv = SAGEConv(in_channels=in_size, out_channels=hidden_size)
            #conv = GINConv(in_channels=in_size, out_channels=hidden_size,nn= MLP(in_size, [hidden_size,hidden_size]))
            
            #mpnn = MPNN(in_channels=in_size, out_channels=hidden_size, in_channels2=in_size2)
            self.convs2.append(conv)


        self.lin_out = Linear(hidden_size, output_size)
    
    def forward(self, x, x1, edge_index, edge_weight=None):
        # Message-passing: transform node features based on neighbors
        
        relu = ReLU(inplace=True)
        for conv in self.convs1:
            x1 = conv(x1, edge_index, edge_weight)
            x1 = torch.tanh(x1)

        x = torch.cat((x, x1), dim=1)
        for conv in self.convs2:
            x = conv(x, edge_index, edge_weight)
            x = relu(x)
            #x = conv(x, edge_index, edge_attr=edge_attr)
        # Decoder: post-process extracted features
        out = self.lin_out(x)
        #out = self.sm(out) #does not work with softmax...?
        return out

