import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.num_layers = len(layer_sizes) - 1
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(self.num_layers)
        ])

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for i in range(self.num_layers - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x)


class MlP_Block(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MlP_Block, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x + residual

class Residual_MLP(nn.Module):
    def __init__(self, num_blocks, input_size, hidden_size):
        super(Residual_MLP, self).__init__()
        self.blocks = nn.ModuleList([MlP_Block(hidden_size, hidden_size) for _ in range(num_blocks)])
        self.final_linear = nn.Linear(hidden_size, 10)
        self.initial_linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.initial_linear(x))
        for block in self.blocks:
            x = block(x)
        x = F.relu(self.final_linear(x))
        return x

class CNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(CNN_Block, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))  

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x

class Transition_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 2):
        super(Transition_Layer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=stride)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return self.pool(x)

class CNN(nn.Module):
    def __init__(self, num_blocks, input_channels, hidden_channels, num_class, dim_input):
        super(CNN, self).__init__()
        self.num_blocks = num_blocks
        blocks = []
        for i in range(len(num_blocks)):
            blocks.append(CNN_Block(hidden_channels[i], hidden_channels[i], num_blocks[i]))
        self.layers = nn.ModuleList(blocks)
        trans = []
        for i in range(1, len(num_blocks)):
            trans.append(Transition_Layer(hidden_channels[i - 1], hidden_channels[i]))
        self.trans = nn.ModuleList(trans)

        dim = dim_input
        for _ in range(len(trans)):
            dim = dim // 2

        self.initial_conv = nn.Conv2d(input_channels, hidden_channels[0], kernel_size=3, padding=1)

        self.fc = nn.Linear(hidden_channels[-1] * (dim) * (dim), num_class)
    def forward(self, x):
        x = F.relu(self.initial_conv(x))
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if (i < len(self.trans)):
                x = self.trans[i](x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Residual_Block(nn.Module):
    def __init__(self, channels):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x + residual

class Block(nn.Module):
    def __init__(self, channels, num_layers):
        super(Block, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(Residual_Block(channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x= F.relu(self.bn(self.conv(x)))
        return self.pool(x)

class ResNet(nn.Module):
    def __init__(self, channels, num_blocks, input_channels, num_class):
        super(ResNet, self).__init__()
        blocks = []
        for i in range(len(num_blocks)):
            blocks.append(Block(channels[i], num_blocks[i]))
        self.layers = nn.ModuleList(blocks)
        self.initial_conv = nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1)
        trans = []
        for i in range(1,len(num_blocks)):
          trans.append(Transition(channels[i-1], channels[i]))
        self.trans = nn.ModuleList(trans)
        dim = 32
        for _ in range(len(trans)):
          dim = dim // 2
        self.fc = nn.Linear(channels[-1] * dim * dim, num_class)

    def forward(self, x):
        x = F.relu(self.initial_conv(x))
        for i in range(len(self.layers)):
          x = self.layers[i](x)
          if (i < len(self.trans)):
            x = self.trans[i](x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CNN_CAM(nn.Module):
    def __init__(self, num_blocks, input_channels, hidden_channels, num_class):
        super(CNN_CAM, self).__init__()
        self.num_blocks = num_blocks
        blocks = []
        for i in range(len(num_blocks)):
            blocks.append(CNN_Block(hidden_channels[i], hidden_channels[i], num_blocks[i]))
        self.layers = nn.ModuleList(blocks)

        trans = []
        for i in range(1, len(num_blocks)):
            trans.append(Transition_Layer(hidden_channels[i - 1], hidden_channels[i], stride=1))
        self.trans = nn.ModuleList(trans)

        self.initial_conv = nn.Conv2d(input_channels, hidden_channels[0], kernel_size=3, padding=1)

        self.fc = nn.Linear(hidden_channels[-1], num_class)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward_features(self, x):
        x = F.relu(self.initial_conv(x))
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < len(self.trans):
                x = self.trans[i](x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.gap(x) 
        x = torch.flatten(x, 1)  
        x = self.fc(x)  
        return x

class CAM():
    def __init__(self, model, type='resnet'):
        self.model = model.eval()
        self.type = type
        self.feature = nn.Sequential(*list(self.model.children())[:-2])
        self.fc_weights = self.model.fc.weight.detach()  

    def generate(self, input):
        with torch.no_grad():
            if(self.type == "resnet"):
                features = self.feature(input)
            else:
                features = self.model.forward_features(input)
       
        output = self.model(input)
        class_idx = torch.argmax(output).item()
        class_weights = self.fc_weights[class_idx]

        features = features.squeeze(0)  # Rimuovi la dimensione del batch
        # Calcola la mappa di attivazione (CAM)
        cam = torch.sum(class_weights[:, None, None] * features, dim=0)  
        
        # Normalizza la CAM
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()
