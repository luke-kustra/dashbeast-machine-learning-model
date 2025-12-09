
import torch.nn as nn

class WorkoutClassifier(nn.Module):
    """
    A simple feedforward neural network for classifying workouts.
    """
    def __init__(self, num_features, num_classes):
        """
        Initializes the model.

        Args:
            num_features (int): The number of input features (6 for our data).
            num_classes (int): The number of output classes (3 for our data).
        """
        super(WorkoutClassifier, self).__init__()
        self.layer1 = nn.Linear(num_features, 32)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(16, num_classes)

    def forward(self, x):
        """
        Performs the forward pass.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        return x
