import torch
import torch.nn as nn

class LinearLayer(nn.Module):
    '''
    Custom linear layer.
    Accepts weight and bias as a parameter.
    '''
    def __init__(self, input_dim, output_dim, params=None, device=None, linear_only=False):
        super().__init__()

        self.linear_only = linear_only
        if params is not None:
            self.weight = params[:input_dim * output_dim].reshape(input_dim, output_dim)
            self.bias = params[input_dim * output_dim:]
        else:
            self.weight = nn.parameter.Parameter(torch.rand((input_dim, output_dim), device=device), requires_grad=True)
            self.bias = nn.parameter.Parameter(torch.rand(output_dim, device=device), requires_grad=True)
            nn.init.kaiming_normal_(self.weight)

        if not linear_only:
            self.norm = nn.LayerNorm(output_dim, elementwise_affine=(params is None), device=device )
            self.a = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x @ self.weight
        x = x + self.bias
        if not self.linear_only:
            x = self.a(self.norm(x))
        return x

class SRN(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4, out_dim=256, params_list=None, device=None):
        super(SRN, self).__init__()

        
        params_list = [None for i in range(num_layers)] if params_list is None else params_list
        layers = [LinearLayer(3, hidden_dim, params_list[0], device)]
        for i in range(1, num_layers - 1):
            layers += [LinearLayer(hidden_dim, hidden_dim, params_list[i], device)]

        layers += [LinearLayer(hidden_dim, out_dim, params_list[-1], device)]
        
        self.layers = nn.ModuleList(layers)


    def _linear_block(self, in_dim, out_dim):
        return [nn.Linear(in_dim, out_dim, bias=True), # no need to add bias due to normalization right afterwards
                             nn.LayerNorm(out_dim), # no parameters for norm layer if coming from hypernetwork
                             nn.ReLU(inplace=True)]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class HyperLayer(nn.Module):
    # Generate parameters for one layer of SRN
    def __init__(self, in_dim, out_dim, num_layers=3):
        super(HyperLayer, self).__init__()

        layers = []

        for i in range(num_layers - 1):
            layers.append( self._linear_block(in_dim, in_dim) )

        layers += [nn.Linear(in_dim, out_dim)]  # No normalization and activation for output

        self.layers = nn.ModuleList(layers)

    def _linear_block(self, in_dim, out_dim):
        return nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), # no need to add bias due to normalization right afterwards
                             nn.LayerNorm(out_dim),
                             nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class HyperNetwork(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4, out_dim=256):
        super(HyperNetwork, self).__init__()

        # Compute number of parameters of SRN at each layer
        srn_num_params = [3 * hidden_dim + hidden_dim]
        for i in range(num_layers - 2):
            srn_num_params.append(hidden_dim * hidden_dim + hidden_dim)
        srn_num_params.append(hidden_dim * out_dim + out_dim)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

        hyperlayers = []
        for n in srn_num_params:
            hyperlayers.append(HyperLayer(256, n))

        self.hyperlayers = nn.ModuleList(hyperlayers)

    def forward(self, z):
        idx = 0
        params_list = []
        for hyperlayer in self.hyperlayers:
            params = hyperlayer(z)
            params_list.append(params)

        return SRN(self.hidden_dim, self.num_layers, self.out_dim, params_list)

# Ray Marching LSTM
class RMLSTM(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=16):
        super(RMLSTM, self).__init__()

        self.lstm = nn.LSTMCell(in_dim, hidden_dim)
        self.linear =  nn.Linear(hidden_dim, 1)

        nn.init.kaiming_normal_(self.lstm.weight_ih)
        nn.init.orthogonal_(self.lstm.weight_hh)
        nn.init.constant_(self.lstm.bias_ih, 0.0)
        nn.init.constant_(self.lstm.bias_hh, 0.0)

    def forward(self, x, hidden_state, cell_state):
        hx, cx = self.lstm(x, (hidden_state, cell_state))
        out = self.linear(hx)
        return out, (hx, cx)

class PixelGenerator(nn.Module):
    def __init__(self, in_dim=256, num_layers=5):
        super(PixelGenerator, self).__init__()

        layers = []

        for i in range(num_layers):
            layers.append( self._linear_block(in_dim, in_dim) )

        layers += [nn.Linear(in_dim, 3)]  # paper does not bound the output pixel values, nn.Sigmoid()]

        self.layers = nn.ModuleList(layers)

    def _linear_block(self, in_dim, out_dim):
        return nn.Sequential(nn.Linear(in_dim, out_dim, bias=True), # no need to add bias due to normalization right afterwards
                             nn.LayerNorm(out_dim),
                             nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
