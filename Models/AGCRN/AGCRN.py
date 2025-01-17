import torch
import torch.nn as nn
from Models.AGCRN.AGCRNCell import AGCRNCell
torch.set_num_threads(4)

class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class AGCRN(nn.Module):
    def __init__(self, agcrnConfig):
        super(AGCRN, self).__init__()
        self.num_node = agcrnConfig['num_nodes']['default']
        self.input_dim = agcrnConfig['input_dim']['default']
        self.hidden_dim = agcrnConfig['rnn_units']['default']
        self.output_dim = agcrnConfig['output_dim']['default']
        self.horizon = agcrnConfig['horizon']['default']
        self.num_layers = agcrnConfig['num_layers']['default']

        self.default_graph = agcrnConfig['default_graph']['default']
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, agcrnConfig['embed_dim']['default']), requires_grad=True)

        self.encoder = AVWDCRNN(agcrnConfig['num_nodes']['default'], agcrnConfig['input_dim']['default'], agcrnConfig['rnn_units']['default'], agcrnConfig['cheb_order']['default'],
                                agcrnConfig['embed_dim']['default'], agcrnConfig['num_layers']['default'])

        #predictor
        self.end_conv = nn.Conv2d(1, agcrnConfig['horizon']['default'] * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.gateMatrix=None
        self.updateMatrix=None

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        self.gateMatrix=self.encoder.dcrnn_cells[0].gateMatrix
        self.updateMatrix=self.encoder.dcrnn_cells[0].updateMatrix
        return output