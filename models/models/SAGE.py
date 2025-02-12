import torch

from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import Sequential, SAGEConv

class GNN_SAGE_Model(torch.nn.Module):

    def __init__(self, hidden_size: int,
                 input_size: int, 
                 output_size: int,
                 num_layers: int):
        super().__init__()
        torch.manual_seed(12345)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size

        self.convs = torch.nn.ModuleList()
        for l in range(num_layers):
            #in_size = dataset_num_features if l == 0 else hidden_size
            in_size = input_size if l == 0 else hidden_size
            in_size2 = hidden_size + output_size if l == 0 else hidden_size*2
            conv = SAGEConv(in_channels=in_size, out_channels=hidden_size)
            
            #mpnn = MPNN(in_channels=in_size, out_channels=hidden_size, in_channels2=in_size2)
            self.convs.append(conv)

        self.lin_out = Linear(hidden_size, output_size)
    
    def forward(self, x, edge_index):
        # Message-passing: transform node features based on neighbors
        relu = ReLU(inplace=True)
        dropout = Dropout(p=0.2,inplace=True)
        #print(edge_index.dtype)
        for conv in self.convs:
            x = conv(x, edge_index)
            #x = dropout(x)
            x = relu(x)
            #x = conv(x, edge_index, edge_attr=edge_attr)
        # Decoder: post-process extracted features
        x = x.to(torch.float32)
        out = self.lin_out(x)
        return out
