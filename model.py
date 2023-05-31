"""
Generator and discriminator used in MetricGAN(+)
Original authors: Szu-Wei Fu 2020
Github repo: https://github.com/speechbrain

Reimplemented: Wooseok Shin
"""
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


def xavier_init_layer(
    in_size, out_size=None, spec_norm=True, layer_type=nn.Linear, **kwargs
):
    "Create a layer with spectral norm, xavier uniform init and zero bias"
    if out_size is None:
        out_size = in_size

    layer = layer_type(in_size, out_size, **kwargs)
    if spec_norm:
        layer = spectral_norm(layer)

    # Perform initialization
    nn.init.xavier_uniform_(layer.weight, gain=1.0)
    nn.init.zeros_(layer.bias)

    return layer


class Learnable_sigmoid(nn.Module):
    def __init__(self, in_features=257):
        super().__init__()
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True  # set requiresGrad to true!

    def forward(self, x):
        return 1.2 * torch.sigmoid(self.slope * x)


class Generator(nn.Module):
    def __init__(self, causal=False):
        super(Generator, self).__init__()
        dim = 200
        self.lstm = nn.LSTM(257, dim, dropout=0.1, num_layers=2, bidirectional=not causal, batch_first=True)    # causal==False -> bidirectional=True
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0
        """
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

        self.LReLU = nn.LeakyReLU(negative_slope=0.3)
        if not causal:
            dim = dim * 2
        self.fc1 = xavier_init_layer(dim, 300, spec_norm=False)
        self.fc2 = xavier_init_layer(300, 257, spec_norm=False)
        
        self.Learnable_sigmoid = Learnable_sigmoid()
        
    def forward(self, x, lengths=None):
        # Pack sequence for LSTM padding
        if lengths is not None:
            x = self.pack_padded_sequence(x, lengths)

        outputs, _ = self.lstm(x)

        # Unpack the packed sequence
        if lengths is not None:
            outputs = self.pad_packed_sequence(outputs)

        outputs = self.fc1(outputs)
        outputs = self.LReLU(outputs)
        outputs = self.fc2(outputs)

        outputs = self.Learnable_sigmoid(outputs)
        return outputs

    def pack_padded_sequence(self, inputs, lengths):
        lengths = lengths.cpu()
        return torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)

    def pad_packed_sequence(self, inputs):
        outputs, lengths = torch.nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True)
        return outputs


class Discriminator(nn.Module):
    def __init__(self, num_target_metrics=1):
        super(Discriminator, self).__init__()

        self.BN = nn.BatchNorm2d(num_features=2, momentum=0.01)

        layers = []
        base_channel = 16
        layers.append(xavier_init_layer(2, base_channel, layer_type=nn.Conv2d, kernel_size=(5,5)))
        layers.append(xavier_init_layer(base_channel, base_channel*2, layer_type=nn.Conv2d, kernel_size=(5,5)))
        layers.append(xavier_init_layer(base_channel*2, base_channel*4, layer_type=nn.Conv2d, kernel_size=(5,5)))
        layers.append(xavier_init_layer(base_channel*4, base_channel*8, layer_type=nn.Conv2d, kernel_size=(5,5)))
        self.layers = nn.ModuleList(layers)
        
        self.LReLU = nn.LeakyReLU(0.3)
        
        self.fc1 = xavier_init_layer(base_channel*8, 50)
        self.fc2 = xavier_init_layer(50, 10)
        self.fc3 = xavier_init_layer(10, num_target_metrics)

    def forward(self, x):
        x = self.BN(x)
        for layer in self.layers:
            x = layer(x)
            x = self.LReLU(x)

        x = torch.mean(x, (2, 3))    # Average Pooling
        x = self.LReLU(self.fc1(x))
        x = self.LReLU(self.fc2(x))

        x = self.fc3(x)
        return x
    
    
