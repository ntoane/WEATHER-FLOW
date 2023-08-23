import torch

class Seq2SeqAttrs:
    def __init__(self, sparse_idx, angle_ratio, geodesic, modelConfig,horizon):
        self.sparse_idx = sparse_idx
        self.max_view = int(modelConfig['max_view']['default'])
        self.cl_decay_steps = int(modelConfig['cl_decay_steps']['default'])
        self.node_num = int(modelConfig['num_nodes']['default'])
        self.layer_num = int(modelConfig['layer_num']['default'])
        self.rnn_units = int(modelConfig['rnn_units']['default'])
        self.input_dim = int(modelConfig['input_dim']['default'])
        self.output_dim = int(modelConfig['output_dim']['default'])
        self.seq_len = int(modelConfig['seq_len']['default'])
        self.lck_structure = modelConfig['lck_structure']['default']
        self.embed_dim = int(modelConfig['embed_dim']['default'])
        self.location_dim = int(modelConfig['location_dim']['default'])
        self.horizon = horizon
        self.hidden_units = int(modelConfig['hidden_units']['default'])
        self.block_num = int(modelConfig['block_num']['default'])
        angle_ratio = torch.sparse.FloatTensor(
            self.sparse_idx, 
            angle_ratio, 
            (self.node_num,self.node_num)
            ).to_dense() 
        self.angle_ratio = angle_ratio + torch.eye(*angle_ratio.shape).to(angle_ratio.device)
        self.geodesic =  torch.sparse.FloatTensor(
            self.sparse_idx, 
            geodesic, 
            (self.node_num,self.node_num)
            ).to_dense()