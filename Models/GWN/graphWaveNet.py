import torch
import torch.nn as nn
import torch.nn.functional as F


""" This class defines a Python class named nconv that is a subclass of nn.Module. 
    It is used to create a neural network layer for performing graph convolution.

    Returns:
        _type_: _description_
"""
class nconv(nn.Module):
    def __init__(self):
        """ The __init__ method is called when an instance of the nconv class is created.
            It initializes the class by calling the __init__ method of its parent class nn.Module.
        """
        super(nconv, self).__init__()

    
    def forward(self, x, A):
        """
        Performs the forward computation of the nconv layer.

        Args:
            x (torch.Tensor): Input tensor.
            A (torch.Tensor): Adjacency matrix.

        Returns:
            torch.Tensor: Output tensor after applying the graph convolution.

        The forward method applies a graph convolution operation to the input tensor x
        using the adjacency matrix A. It uses the torch.einsum function to perform
        element-wise multiplication and reduction operation between x and A,
        followed by making the resulting tensor contiguous for efficient computations.
        """
        
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        """
        Initializes the linear class, a subclass of nn.Module.

        Args:
            c_in (int): Number of input channels.
            c_out (int): Number of output channels.
        """
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        """
        Performs the forward computation of the linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the linear transformation.

        The forward method applies the linear transformation to the input tensor x
        using the convolutional layer defined in the mlp attribute.
        """
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        """
        Initializes the gcn class, a subclass of nn.Module.

        Args:
            c_in (int): Number of input channels.
            c_out (int): Number of output channels.
            dropout (float): Dropout probability.
            support_len (int): Number of support matrices.
            order (int): Order of graph convolution.
        """
        super(gcn, self).__init__()
        self.nconv = nconv()
        # The nconv attribute represents an instance of the nconv class,
        # which is responsible for performing graph convolutions.
        
        c_in = (order * support_len + 1) * c_in
        # Update the number of input channels based on the order of graph convolution and 
        # the number of support matrices.
        
        self.mlp = linear(c_in, c_out)
        # The mlp attribute represents an instance of the linear class,
        # which is a linear transformation layer applied after the graph convolutions.

        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        """
        Performs the forward computation of the gcn model.

        Args:
            x (torch.Tensor): Input tensor.
            support (list of torch.Tensor): List of support matrices.

        Returns:
            torch.Tensor: Output tensor after applying the graph convolutional network.

        The forward method applies the graph convolutional network to the input tensor x.
        It performs graph convolutions using the support matrices in the provided list,
        followed by applying a linear transformation and dropout to the resulting tensor.
        """
        out = [x]
        # Initialize a list to store intermediate output tensors.
        # The first tensor in the list is the input tensor x.
        for a in support:
            x1 = self.nconv(x, a)
            # Perform graph convolution using the nconv module and the current support matrix a.
            out.append(x1)
            # Append the resulting tensor to the list of intermediate outputs.
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                # Perform additional graph convolutions based on the order.
                out.append(x2)
                x1 = x2
                 # Update the current tensor for the next iteration.

        h = torch.cat(out, dim=1)
        # Concatenate the intermediate output tensors along the channel dimension.

        h = self.mlp(h)
        # Apply the linear transformation to the concatenated tensor.
        
        h = F.dropout(h, self.dropout, training=self.training)
        # Apply dropout to the tensor to prevent overfitting.
        return h


class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None,
                 in_dim=6, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2):
        """
        Initializes the gwnet class, a subclass of nn.Module.

        Args:
            device: The device to run the model on.
            num_nodes (int): Number of nodes in the graph.
            dropout (float): Dropout probability.
            supports: List of support matrices for graph convolutions.
            gcn_bool (bool): Flag to enable/disable graph convolutions.
            addaptadj (bool): Flag to enable/disable adaptive adjacency matrix.
            aptinit: Initial adjacency matrix for adaptive adjacency.
            in_dim (int): Number of input dimensions.
            out_dim (int): Number of output dimensions.
            residual_channels (int): Number of channels for residual connections.
            dilation_channels (int): Number of channels for dilation convolutions.
            skip_channels (int): Number of channels for skip connections.
            end_channels (int): Number of channels for end convolutions.
            kernel_size (int): Kernel size for dilated convolutions.
            blocks (int): Number of blocks in the WaveNet architecture.
            layers (int): Number of layers in each block.
        """
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):
        """
        Forward pass of the gwnet model.

        Args:
            input: Input tensor.

        Returns:
            Output tensor after passing through the model.
        """
        in_len = input.size(3)
        
        # Padding the input if its length is smaller than the receptive field
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            # (dilation, init_dilation) = self.dilations[i]

            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                # Graph convolution (if enabled)
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]  # Residual connection

            x = self.bn[i](x) # Batch normalization

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
