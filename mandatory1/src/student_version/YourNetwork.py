import torch
import torch.nn as nn
from RainforestDataset import get_classes_list

class TwoNetworks(nn.Module):
    '''
    This class takes two pretrained networks,
    concatenates the high-level features before feeding these into
    a linear layer.

    functions: forward
    '''
    def __init__(self, pretrained_net1, pretrained_net2):
        super(TwoNetworks, self).__init__()

        _, num_classes = get_classes_list()

        # TODO select all parts of the two pretrained networks, except for
        # the last linear layer.
        
        self.fully_conv1 = nn.Sequential(*list(pretrained_net1.children()))[:-1]    # Selecting all but the last layer of pretraind ResNet18

        ############################

        current_weights = pretrained_net2.conv1.weight           # Extracting weights of first convolutional layer in default pretrained network (ResNet 18)

        out_channels    = pretrained_net2.conv1.out_channels     # Extracting number of output channels of first convolutional layer in default pretrained network (ResNet 18)

        nout, nin, nx, ny = current_weights.shape               
        new_weights = torch.zeros((nout, 1, nx, ny))            # Setting up zeros tensor for new weights to accomodate for the IR channel in the input images            

        nn.init.kaiming_uniform_(new_weights)  # Kaiming He initialization of zeros weights tensor

        pretrained_net2.conv1 = nn.Conv2d(1, out_channels, 
                                        kernel_size = pretrained_net2.conv1.kernel_size,
                                        stride = pretrained_net2.conv1.stride,
                                        padding = pretrained_net2.conv1.padding,
                                        bias = pretrained_net2.conv1.bias)                       # Overwriting current first convolutional layer

        pretrained_net2.conv1.weight = torch.nn.Parameter(new_weights)                           # Setting new weight parameters

        self.fully_conv2 = nn.Sequential(*list(pretrained_net2.children()))[:-1]                 # Selecting all but the last layer of pretraind ResNet18 with modified initial convolutional layer (for IR image)


        # TODO create a linear layer that has in_channels equal to
        # the number of in_features from both networks summed together.
        outfeatures1 = pretrained_net1.fc.in_features
        outfeatures2 = pretrained_net2.fc.in_features
        self.linear = nn.Linear(outfeatures1 + outfeatures2, num_classes)       # Double number of output features since we concatinate RGB and IR results before fully conenncted layer
        self.flatten = nn.Flatten()

    def forward(self, inputs1, inputs2):
        # TODO feed the inputs through the fully convolutional parts
        # of the two networks that you initialised above, and then
        # concatenate the features before the linear layer.
        # And return the result.
        out1 = self.flatten(self.fully_conv1(inputs1))      # Flattening output before input to linear layer
        out2 = self.flatten(self.fully_conv2(inputs2))

        out = torch.cat((out1, out2), 1)
        out = self.linear(out)

        return out

class SingleNetwork(nn.Module):
    '''
    This class takes one pretrained network,
    the first conv layer can be modified to take an extra channel.

    functions: forward
    '''

    def __init__(self, pretrained_net, weight_init = None):
        super(SingleNetwork, self).__init__()

        _, num_classes = get_classes_list()

        self.weight_init = weight_init

        if weight_init is not None:
            # TODO Here we want an additional channel in the weights tensor, specifically in the first
            # conv2d layer so that there are weights for the infrared channel in the input aswell.
            
            ############################
            current_weights = pretrained_net.conv1.weight           # Extracting weights of first convolutional layer in default pretrained network (ResNet 18)

            out_channels    = pretrained_net.conv1.out_channels     # Extracting number of output channels of first convolutional layer in default pretrained network (ResNet 18)

            nout, nin, nx, ny = current_weights.shape               
            new_weights = torch.zeros((nout, 4, nx, ny))            # Setting up zeros tensor for new weights to accomodate for the IR channel in the input images            


            ############################


            if weight_init == "kaiminghe":
                
                ###########################

                nn.init.kaiming_uniform_(new_weights[-1, ...])  # Kaiming He initialization of zeros weights tensor

                ###########################

            # TODO Create a new conv2d layer, and set the weights to be
            # what you created above. You will need to pass the weights to
            # torch.nn.Parameter() so that the weights are considered
            # a model parameter.
            # eg. first_conv_layer.weight = torch.nn.Parameter(your_new_weights)

            ###########################

            new_weights[:, :-1, :, :] = current_weights     # Copying RBG channel weights to new weights tensor

            pretrained_net.conv1 = nn.Conv2d(4, out_channels, 
                                            kernel_size = pretrained_net.conv1.kernel_size,
                                            stride = pretrained_net.conv1.stride,
                                            padding = pretrained_net.conv1.padding,
                                            bias = pretrained_net.conv1.bias)                       # Overwriting current first convolutional layer

            pretrained_net.conv1.weight = torch.nn.Parameter(new_weights)                           # Setting new weight parameters

            ###########################


        # TODO Overwrite the last linear layer.
        
        #############################

        outfeatures = pretrained_net.fc.in_features
        pretrained_net.fc = nn.Linear(outfeatures, 17, bias = True)      # Overwriting last fully connected layer to accomodate for 17 output classes 
        #############################

        self.net = pretrained_net

    def forward(self, inputs):
        return self.net(inputs)





