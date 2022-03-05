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
    def __init__(self, pretrained_net1, pretrained_net2, freeze = False):
        """Initiating TwoNetworks class.

        Parameters
        ----------
        pretrained_net1 : nn.Module
            Instance of pretrained model; ResNet18. Will be the base of RGB branch of 
            TwoNetworks.
        pretrained_net2 : nn.Module
            Instance of pretrained model; ResNet18. Will be the base of Ir branch of 
            TwoNetworks.
        freeze : bool, optional
            Whether to freeze middle layers of pretrained models, by default False.
            This is only used for debugging, but not used in "production runs".
        """
        super(TwoNetworks, self).__init__()

        _, num_classes = get_classes_list() # Get number of classes to classify


        self.fully_conv1 = nn.Sequential(*list(pretrained_net1.children()))[:-1]    # Selecting all but the last layer of pretraind ResNet18

        current_weights = pretrained_net2.conv1.weight           # Extracting weights of first convolutional layer in default pretrained network (ResNet 18)

        out_channels    = pretrained_net2.conv1.out_channels     # Extracting number of output channels of first convolutional layer in default pretrained network (ResNet 18)

        nout, nin, nx, ny = current_weights.shape               
        new_weights = torch.zeros((nout, 1, nx, ny))            # Setting up zeros tensor for new weights to accomodate for the IR channel in the input images            

        nn.init.kaiming_normal_(new_weights, nonlinearity = "relu")  # Kaiming He initialization of zeros weights tensor. 
                                                                     # Using ReLU non-linearity since that is the activation used 
                                                                     # in ResNet18

        pretrained_net2.conv1 = nn.Conv2d(1, out_channels, 
                                        kernel_size = pretrained_net2.conv1.kernel_size,
                                        stride = pretrained_net2.conv1.stride,
                                        padding = pretrained_net2.conv1.padding,
                                        bias = pretrained_net2.conv1.bias)                       # Overwriting current first convolutional layer

        pretrained_net2.conv1.weight = torch.nn.Parameter(new_weights)                           # Setting new weight parameters

        self.fully_conv2 = nn.Sequential(*list(pretrained_net2.children()))[:-1]                 # Selecting all but the last layer of pretraind ResNet18 with modified initial convolutional layer (for IR image)



        outfeatures1 = pretrained_net1.fc.in_features
        outfeatures2 = pretrained_net2.fc.in_features

        self.linear = nn.Linear(outfeatures1 + outfeatures2, num_classes)       # Combined number of output features since we concatinate 
                                                                                # RGB and IR results before fully conenncted layer
        
        if freeze:  # Whether to freeze intermediate layers of pretrained branches. Not used for "production runs" only for debugging
            for param in list(self.fully_conv1.parameters())[1:]:
                param.requires_grad = False
            for param in list(self.fully_conv2.parameters())[1:]:
                param.requires_grad = False

        self.flatten = nn.Flatten() # Flattening layer to flatten output of the two convolutional branches.

    def forward(self, inputs1, inputs2):
        """Forward function of TwoNetworks class, which is called when calling instance 
           of the class.

        Parameters
        ----------
        inputs1 : torch.tensor
            RGB image tensor.
        inputs2 : torch.tensor
            IR image tensor.

        Returns
        -------
        torch.tensor
            Raw model output. When put through sigmoid the output would correspond to
            class probability scores.
        """
        out1 = self.flatten(self.fully_conv1(inputs1))      # Flattening output before input to linear layer
        out2 = self.flatten(self.fully_conv2(inputs2))
       
        out = torch.cat((out1, out2), 1)                    # Concatinate RGB and Ir branch outputs
        out = self.linear(out)

        return out

class SingleNetwork(nn.Module):
    '''
    This class takes one pretrained network,
    the first conv layer can be modified to take an extra channel.

    functions: forward
    '''

    def __init__(self, pretrained_net, weight_init = None, freeze = False):
        """Initiating SingleNetwork class.

        Parameters
        ----------
        pretrained_net : nn.Module
            Pretrained model to base SingleNetwork on; ResNet18.
        weight_init : str, optional
            Whether to use SingleNetwork for RGB and Ir, and initialize Ir weights
            of first convolutional layer by Kaiming normal distribution. By default None
        freeze : bool, optional
            Whether to freeze middle layers of pretrained models, by default False.
            This is only used for debugging, but not used in "production runs".
        """
        
        super(SingleNetwork, self).__init__()

        _, num_classes = get_classes_list()     # Get number of classes

        self.weight_init = weight_init

        if weight_init is not None:
            # Using SingleNetwork for RGB and Ir image classification if weight_init is not None.

            current_weights = pretrained_net.conv1.weight           # Extracting weights of first convolutional layer in default pretrained network (ResNet 18)

            out_channels    = pretrained_net.conv1.out_channels     # Extracting number of output channels of first convolutional layer in default pretrained network (ResNet 18)

            nout, nin, nx, ny = current_weights.shape               
            new_weights = torch.empty((nout, 4, nx, ny))            # Setting up zeros tensor for new weights to accomodate for the IR channel in the input images            

            if weight_init == "kaiminghe":
                nn.init.kaiming_normal_(new_weights, nonlinearity = "relu")  # Kaiming He initialization of zeros weights tensor.
                                                                             # Since ResNet18 base uses ReLU activations we use "relu"
                                                                             # non-linearity

            new_weights[:, :3, :, :] = current_weights     # Copying RBG channel weights to new weights tensor

            pretrained_net.conv1 = nn.Conv2d(4, out_channels, 
                                            kernel_size = pretrained_net.conv1.kernel_size,
                                            stride = pretrained_net.conv1.stride,
                                            padding = pretrained_net.conv1.padding,
                                            bias = pretrained_net.conv1.bias)                       # Overwriting current first convolutional layer

            pretrained_net.conv1.weight = torch.nn.Parameter(new_weights)                           # Setting new weight parameters

        if freeze:  # Whether to freeze intermediate layers of pretrained branches. Not used for "production runs" only for debugging
            for param in list(pretrained_net.parameters())[1:-1]:
                param.requires_grad = False

        outfeatures = pretrained_net.fc.in_features
        pretrained_net.fc = nn.Linear(outfeatures, num_classes, bias = True)      # Overwriting last fully connected layer to accomodate for 17 output classes 
        
        self.net = pretrained_net

    def forward(self, inputs):
        """Forward function of SingleNetwork class, which is called when calling instance 
           of the class.


        Parameters
        ----------
        inputs : torch.tensor
            Image tensor. Can contain both RGB and Ir of just RGB, depending on
            whether weights_init is given in initiating class instance.

        Returns
        -------
        torch.tensor
            Raw model output. When put through sigmoid the output would correspond to
            class probability scores.
        """
        return self.net(inputs)





