import torch
import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Remove the fully connected layers

    def forward(self, x):
        x = self.features(x)
        return x

# Set device to CPU
device = torch.device('cpu')

# Example usage
feature_extractor = FeatureExtractor().to(device)

# Example batch of images
dataiter = iter(trainloader)
images, labels = next(dataiter)
images = images.to(device)

# Extract feature maps
feature_maps = feature_extractor(images)
print("Feature maps shape:", feature_maps.shape)



#//////



class HierarchicalAttention(nn.Module):
    def __init__(self, in_channels):
        super(HierarchicalAttention, self).__init__()
        self.scale1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.scale2 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1)
        self.attention1 = nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        self.attention2 = nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        
    def forward(self, x):
        scale1_feat = self.scale1(x)
        scale2_feat = self.scale2(scale1_feat)
        attention1 = torch.sigmoid(self.attention1(scale1_feat))
        attention2 = torch.sigmoid(self.attention2(scale2_feat))
        hierarchical_attention = torch.cat([attention1, attention2], dim=1)
        return hierarchical_attention

# Example usage
hierarchical_attention_module = HierarchicalAttention(in_channels=512).to(device)
hierarchical_attention = hierarchical_attention_module(feature_maps)
print("Hierarchical attention shape:", hierarchical_attention.shape)



#//////


class ContextualRelevance(nn.Module):
    def __init__(self, in_channels):
        super(ContextualRelevance, self).__init__()
        self.context_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        contextual_relevance = torch.sigmoid(self.context_conv(x))
        return contextual_relevance

# Example usage
contextual_relevance_module = ContextualRelevance(in_channels=512).to(device)
contextual_relevance = contextual_relevance_module(feature_maps)
print("Contextual relevance shape:", contextual_relevance.shape)



#////////



class TemporalMemory(nn.Module):
    def __init__(self, memory_size, feature_size):
        super(TemporalMemory, self).__init__()
        self.memory_size = memory_size
        self.memory = torch.zeros(memory_size, feature_size).to(device)
        
    def forward(self, x):
        b, c, h, w = x.size()
        current_features = x.view(b, c, -1).mean(dim=2)
        differences = torch.cdist(current_features, self.memory)
        temporal_novelty = torch.mean(differences, dim=1).view(b, 1, 1, 1)
        self.memory = torch.cat([self.memory[1:], current_features.detach()], dim=0)
        return temporal_novelty

# Example usage
temporal_memory_module = TemporalMemory(memory_size=100, feature_size=512).to(device)
temporal_novelty = temporal_memory_module(feature_maps)
print("Temporal novelty shape:", temporal_novelty.shape)




#//////




class TemporalMemory(nn.Module):
    def __init__(self, memory_size, feature_size):
        super(TemporalMemory, self).__init__()
        self.memory_size = memory_size
        self.memory = torch.zeros(memory_size, feature_size).to(device)
        
    def forward(self, x):
        b, c, h, w = x.size()
        current_features = x.view(b, c, -1).mean(dim=2)
        differences = torch.cdist(current_features, self.memory)
        temporal_novelty = torch.mean(differences, dim=1).view(b, 1, 1, 1)
        self.memory = torch.cat([self.memory[1:], current_features.detach()], dim=0)
        return temporal_novelty

# Example usage
temporal_memory_module = TemporalMemory(memory_size=100, feature_size=512).to(device)
temporal_novelty = temporal_memory_module(feature_maps)
print("Temporal novelty shape:", temporal_novelty.shape)






#                       ***************************             #

