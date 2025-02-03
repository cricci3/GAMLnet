import torch

from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


def random_walk_matrix(edge_index, num_nodes: int = None):
    source, target = edge_index[0], edge_index[1]
    in_deg = degree(target, num_nodes=num_nodes)   # D
    edge_weight = 1 / in_deg[target]               # D^-1 A
    return edge_index, edge_weight


class MPNN(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")                         # "sum" aggregation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin_message = Linear(in_channels, out_channels, # weights 𝚯_1
                                  bias=False)
        self.lin_update = Linear(in_channels, out_channels,  # weights 𝚯_2
                                 bias=True)                  # the bias vector 𝐛

    def forward(self, x, edge_index, edge_weight=None):
        # 0. if GSO not already computed, compute it here
        if edge_weight is None:
          _, edge_weight = random_walk_matrix(edge_index)
        # 1. m_j→𝑖 = x_j𝚯_1
        m_ji = self.lin_message(x)  # we can project here with isotropic GNNs
        # 2. m_𝑖 = add(ã_ji ⋅ m_j→𝑖)_j∈𝑁(i)
        m_i = self.propagate(edge_index, m=m_ji, edge_weight=edge_weight)
        # 3. h_𝑖 = tanh(x_i𝚯_2 + m_i + 𝐛)
        h_i = torch.tanh(self.lin_update(x) + m_i)
        return h_i

    def message(self, m_j, edge_weight):
        return edge_weight.view(-1, 1) * m_j  # ã_ji ⋅ m_j→𝑖


class GNN_MPNN_Model(torch.nn.Module):

    def __init__(self, hidden_size: int,
                 num_layers: int, 
                 input_size: int, 
                 output_size: int):
        super().__init__()
        torch.manual_seed(12345)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size

        self.mpnns = torch.nn.ModuleList()
        for l in range(num_layers):
            in_size = input_size if l == 0 else hidden_size
            mpnn = MPNN(in_channels=in_size, out_channels=hidden_size)
            self.mpnns.append(mpnn)

        self.lin_out = Linear(hidden_size, output_size)
    
    def forward(self, x, edge_index, edge_weight=None):
        # Message-passing: transform node features based on neighbors
        for mpnn in self.mpnns:
            x = mpnn(x, edge_index, edge_weight)
        # Decoder: post-process extracted features
        out = self.lin_out(x)
        return out
