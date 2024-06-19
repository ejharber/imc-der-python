import torch
from torch import nn
import numpy as np

def data2imgs(data_traj, data_force):
    img_pos = data_traj
    img_pos_x = np.expand_dims(img_pos[:, ::2, :], 1)
    img_pos_y = np.expand_dims(img_pos[:, 1::2, :], 1)
    img_pos = np.concatenate((img_pos_x, img_pos_y), axis=1)
    img_pos = np.expand_dims(img_pos, 1)

    img_force = data_force
    img_force_x = np.expand_dims(img_force[:, ::2, :], 1)
    img_force_y = np.expand_dims(img_force[:, 1::2, :], 1)
    img_force = np.concatenate((img_force_x, img_force_y), axis=1)

    print(img_pos.shape, img_force.shape)

    return img_pos, img_force


def data2imgs_test(data_traj, data_force):
    img_pos = data_traj
    img_pos_x = np.expand_dims(img_pos[::2, :], 0)
    img_pos_y = np.expand_dims(img_pos[1::2, :], 0)
    img_pos = np.concatenate((img_pos_x, img_pos_y), axis=0)
    img_pos = np.expand_dims(img_pos, 0)
    img_pos = np.repeat(img_pos, 100_000, 0)
    img_pos = torch.from_numpy(img_pos.astype(np.float32))

    img_force = data_force
    img_force_x = np.expand_dims(img_force[::2, :], 0)
    img_force_y = np.expand_dims(img_force[1::2, :], 0)
    img_force = np.concatenate((img_force_x, img_force_y), axis=0)
    img_force = np.expand_dims(img_force, 0)
    img_force = np.repeat(img_force, 100_000, 0)
    img_force = torch.from_numpy(img_force.astype(np.float32))

    return img_pos, img_force

# def data2imgs(data_traj, data_force):
#     img_pos = data_traj
#     img_pos_x = np.expand_dims(img_pos[:, ::2, :], 1)
#     img_pos_y = np.expand_dims(img_pos[:, 1::2, :], 1)
#     img_pos = np.concatenate((img_pos_x, img_pos_y), axis=1)
#     img_pos = np.expand_dims(img_pos[:, 1::2, :], 1)

#     img_force = data_force
#     img_force_x = np.expand_dims(img_force[:, ::2, :], 1)
#     img_force_y = np.expand_dims(img_force[:, 1::2, :], 1)
#     img_force = np.concatenate((img_force_x, img_force_y), axis=1)

#     print(img_pos.shape, img_force.shape)

#     return img_pos, img_force

class IterativeNeuralNetwork(nn.Module):
    class PrintLayer(nn.Module):
        def __init__(self):
            super(PrintLayer, self).__init__()
                        
        def forward(self, x):
            # Do your print / debug stuff here
            print(x.shape)
            return x



    def __init__(self, include_force = True): 

        def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer

        super(IterativeNeuralNetwork, self).__init__()
        self.include_force = include_force

        self.cnn_pos = nn.Sequential(
            layer_init(nn.Conv2d(2, 4, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(4, 4, 2, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(4, 4, 2, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(4, 1, 2, stride=1)),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            # layer_init(nn.Linear(11, 256)),
            # nn.ReLU(),

        )

        self.cnn_force = nn.Sequential(
            layer_init(nn.Conv1d(2, 4, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv1d(4, 4, 2, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv1d(4, 4, 2, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv1d(4, 1, 2, stride=1)),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            # layer_init(nn.Linear(11, 256)),
            # nn.ReLU(),
        )

        if self.include_force:
            self.network = nn.Sequential(
                layer_init(nn.Linear(11 + 11 + 3, 256)),
                nn.ReLU(),
                layer_init(nn.Linear(256, 256)),
                nn.ReLU(),
                layer_init(nn.Linear(256, 2)),
                )
        else: 
            self.network = nn.Sequential(
                layer_init(nn.Linear(11 + 3, 256)),
                nn.ReLU(),
                layer_init(nn.Linear(256, 256)),
                nn.ReLU(),
                layer_init(nn.Linear(256, 2)),
                )

        
    def forward(self, img_pos, img_force, next_action):
        img_force = img_force[:,:,0,:]
        x_pos = self.cnn_pos(img_pos)
        if self.include_force:
            x_force = self.cnn_force(img_force)
            x = torch.cat((x_pos, x_force, next_action), dim=1)
        else:
            x = torch.cat((x_pos, next_action), dim=1)
        x = self.network(x)
        return x

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
                    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

class IterativeNeuralNetwork_2(nn.Module):




    def __init__(self, include_force = True): 


        def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer

        super(IterativeNeuralNetwork_2, self).__init__()
        self.include_force = include_force

        self.cnn_pos = nn.Sequential(
            PrintLayer(),
            layer_init(nn.Conv2d(2, 8, 4, stride=2)),
            nn.ReLU(),
            PrintLayer(),
            layer_init(nn.Conv2d(8, 16, 2, stride=2)),
            nn.ReLU(),
            PrintLayer(),
            layer_init(nn.Conv2d(16, 32, 2, stride=2)),
            nn.ReLU(),
            PrintLayer(),
            layer_init(nn.Conv2d(32, 4, 2, stride=1)),
            nn.ReLU(),
            PrintLayer(),
            nn.Flatten(start_dim=1),
            PrintLayer(),
            # layer_init(nn.Linear(11, 256)),
            # nn.ReLU(),
        )

        self.cnn_force = nn.Sequential(
            layer_init(nn.Conv1d(2, 8, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv1d(8, 16, 2, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv1d(16, 32, 2, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv1d(32, 4, 2, stride=1)),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            # layer_init(nn.Linear(11, 256)),
            # nn.ReLU(),
        )

        if self.include_force:
            self.network = nn.Sequential(
                layer_init(nn.Linear(90, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 3)),
                )
        else: 
            self.network = nn.Sequential(
                layer_init(nn.Linear(47 + 1, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 3)),
                )

        
    def forward(self, img_pos, img_force, next_action):
        img_force = img_force[:,:,0,:]
        img_pos = img_pos[:,0,:,:, :]

        # print(img_pos.shape, img_force.shape, next_action.shape)

        x_pos = self.cnn_pos(img_pos)



        if self.include_force:
            x_force = self.cnn_force(img_force)

            print(x_pos.shape, x_force.shape, next_action.shape)
            x = torch.cat((x_pos, x_force, next_action), dim=1)
        else:
            x = torch.cat((x_pos, next_action), dim=1)
        x = self.network(x)
        return x

class IterativeNeuralNetwork3D(nn.Module):
    def __init__(self, include_force = False): 

        def layer_init(layer):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.constant_(layer.bias, 0)
            return layer

        super(IterativeNeuralNetwork, self).__init__()
        self.include_force = include_force

        self.cnn_pos = nn.Sequential(
            layer_init(nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(2, 4, 4), stride=2, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv3d(in_channels=4, out_channels=1, kernel_size=(2, 4, 4), stride=2)),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=2),
            layer_init(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(4, 4), stride=2)),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=2),
            layer_init(nn.Conv1d(1, 1, 4, stride=2)),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            # layer_init(nn.Linear(11, 256)),
            # nn.ReLU(),
        )

        self.cnn_force = nn.Sequential(
            layer_init(nn.Conv1d(2, 4, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv1d(4, 4, 2, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv1d(4, 4, 2, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv1d(4, 1, 2, stride=1)),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            # layer_init(nn.Linear(11, 256)),
            # nn.ReLU(),
        )

        if self.include_force:
            self.network = nn.Sequential(
                layer_init(nn.Linear(7, 64)),
                nn.ReLU(),
                layer_init(nn.Linear(64, 64)),
                nn.ReLU(),
                layer_init(nn.Linear(64, 2)),
                )
        else: 
            self.network = nn.Sequential(
                layer_init(nn.Linear(7, 64)),
                nn.ReLU(),
                layer_init(nn.Linear(64, 64)),
                nn.ReLU(),
                layer_init(nn.Linear(64, 2)),
                )

        
    def forward(self, img_pos, img_force, next_action):
        img_force = img_force[:,:,0,:]
        x_pos = self.cnn_pos(img_pos)
        if self.include_force:
            x_force = self.cnn_force(img_force)
            x = torch.cat((x_pos, x_force, next_action), dim=1)
        else:
            x = torch.cat((x_pos, next_action), dim=1)
        x = self.network(x)
        return x