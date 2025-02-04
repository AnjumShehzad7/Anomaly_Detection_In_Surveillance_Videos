import torch  # PyTorch library for deep learning
import torch.nn as nn  # Neural network modules
from torch.nn import functional as F  # Functional API for operations

class Learner(nn.Module):
    """
    A neural network model for binary classification.
    
    Args:
        input_dim (int): Dimensionality of the input features.
        drop_p (float): Dropout probability.
    """
    def __init__(self, input_dim=2048, drop_p=0.0):
        super(Learner, self).__init__()
        
        # Define the classifier network using sequential layers
        # This consists of multiple fully connected (linear) layers with ReLU activations and dropout for regularization.
        # The network follows a structure of:
        # - Input layer (size: input_dim)
        # - Hidden layer 1 (size: 512, activation: ReLU, dropout: 0.6)
        # - Hidden layer 2 (size: 32, activation: ReLU, dropout: 0.6)
        # - Output layer (size: 1, activation: Sigmoid)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),  # Fully connected layer
            nn.ReLU(),  # Activation function
            nn.Dropout(0.6),  # Dropout for regularization
            nn.Linear(512, 32),  # Fully connected layer
            nn.ReLU(),  # Activation function
            nn.Dropout(0.6),  # Dropout for regularization
            nn.Linear(32, 1),  # Final output layer
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )
        
        self.drop_p = 0.6  # Dropout probability
        self.weight_init()  # Initialize weights
        self.vars = nn.ParameterList()  # Store model parameters manually

        # Store parameters in the parameter list
        for i, param in enumerate(self.classifier.parameters()):
            self.vars.append(param)

    def weight_init(self):
        """
        Initializes weights of linear layers using Xavier Normal initialization.
        """
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x, vars=None):
        """
        Forward pass of the model.
        
        Args:
            x (Tensor): Input tensor.
            vars (list, optional): Custom parameters for meta-learning. Defaults to None.
        
        Returns:
            Tensor: Output probabilities after sigmoid activation.
        """
        if vars is None:
            vars = self.vars
        
        x = F.linear(x, vars[0], vars[1])  # First linear layer
        x = F.relu(x)  # Activation function
        x = F.dropout(x, self.drop_p, training=self.training)  # Dropout
        
        x = F.linear(x, vars[2], vars[3])  # Second linear layer
        x = F.dropout(x, self.drop_p, training=self.training)  # Dropout
        
        x = F.linear(x, vars[4], vars[5])  # Final linear layer
        return torch.sigmoid(x)  # Apply sigmoid activation

    def parameters(self):
        """
        Override this function to return the manually stored parameters.
        
        Returns:
            nn.ParameterList: List of model parameters.
        """
        return self.vars
