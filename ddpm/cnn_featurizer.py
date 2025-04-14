import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights

class Inception3DEncoder(nn.Module):
    def __init__(self, use_pretrained=True):
        super(Inception3DEncoder, self).__init__()
        if use_pretrained:
            # Use the new weights API; the default weights include aux_logits=True
            weights = Inception_V3_Weights.IMAGENET1K_V1
            self.inception = inception_v3(weights=weights)
        else:
            self.inception = inception_v3(weights=None)
            
        # Replace the final fully connected layer with an identity to obtain a feature vector.
        # The Inception v3 model normally outputs a 2048-dim vector.
        self.inception.fc = nn.Identity()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): A tensor of shape (batch, depth, height, width)
                              For example, (B, 16, 128, 128)
        Returns:
            torch.Tensor: A tensor of shape (batch, feature_dim)
                          where feature_dim is 2048.
        """
        batch_size, depth, H, W = x.shape
        
        # Reshape so that each slice is processed as a separate image.
        # x now has shape (batch*depth, 1, H, W)
        x = x.view(batch_size * depth, 1, H, W)
        
        # Replicate the single channel to make it 3 channels (Inception expects 3-channel images).
        x = x.repeat(1, 3, 1, 1)
        
        # Resize each slice to 299x299 (the input size expected by Inception v3)
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize using ImageNet statistics.
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        # Ensure the model is in evaluation mode to disable the auxiliary branch.
        self.inception.eval()
        with torch.no_grad():
            features = self.inception(x)  # shape: (batch*depth, feature_dim)
        
        # Reshape back to (batch, depth, feature_dim) and average over the depth dimension.
        feature_dim = features.shape[-1]
        features = features.view(batch_size, depth, feature_dim).mean(dim=1)
        
        return features

# Example usage:
if __name__ == '__main__':
    # Create an instance of the encoder.
    model = Inception3DEncoder(use_pretrained=True)
    model.eval()  # Set to evaluation mode
    
    # Create a dummy batch of 3D images: batch_size=2, depth=16, height=128, width=128.
    dummy_input = torch.randn(2, 16, 128, 128)
    
    # Get the vector representations.
    output = model(dummy_input)
    print("Output feature shape:", output.shape)  # Expected: (2, 2048)
