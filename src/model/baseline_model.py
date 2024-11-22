from torch import nn
from torch.nn import Sequential

class CNN_Model(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_Model, self).__init__()
        
        # Define convolutional layers
        #in channels is the 3 RGB channels. Out_channels are 16, it means 16 filters. 
        # And each filter is applied to each of the 3 RGB channels and them summed up.
        # Shape: (num_patches, 3, 16, 16) -> (num_patches, 16, 16, 16)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)  # SAME padding (=1) => no size reduction for patches
        self.relu1 = nn.ReLU()  # ReLU after conv1
        # Shape: (num_patches, 16, 16, 16) -> (num_patches, 32, 16, 16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # SAME padding (=1) => no size reduction for patches
        self.relu2 = nn.ReLU()
        # Max pooling layer
        # Shape: (num_patches, 32, 16, 16) -> (num_patches, 32, 8, 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu3 = nn.ReLU()
        #(num_patches, 32, 8, 8) -> (num_patches, 32, 1, 1)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.conv_final = nn.Conv2d(32, num_classes, kernel_size=1)
        self.flatten = nn.Flatten()
        


    def forward(self, data_object, **batch):
        # First convolution, ReLU
        data_object = self.relu1(self.conv1(data_object))  # Conv1 -> ReLU -> MaxPool
        
        # Second convolution, ReLU, and pooling
        data_object = self.pool(self.relu2(self.conv2(data_object)))  # Conv2 -> ReLU -> MaxPool
        
        data_object = self.pool2(self.relu3(data_object))
        data_object = self.conv_final(data_object)
        data_object = self.flatten(data_object)
        
        return {"logits": data_object}
    
    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
