from torchinfo import summary
from networks.resnet_big import SupConResNet

import torch
import torchprofile
import timm

# Load the model
#model = timm.create_model('resnet50', pretrained=True).cuda()
# Print the summary
#summary(model, (1, 3, 224, 224))
#print(f"ABOVE RESULTS TEST")

model = SupConResNet(name='resnet50timm').cuda()
summary(model, (1, 3, 224, 224))

# Measure FLOPs
flops = torchprofile.profile_macs(model, torch.randn(1, 3, 224, 224).cuda())
print(f"ResNet-50 FLOPs: {flops}")

model = SupConResNet(name='seresnext50timm').cuda()
summary(model, (1, 3, 224, 224))

# Measure FLOPs
flops = torchprofile.profile_macs(model, torch.randn(1, 3, 224, 224).cuda())
print(f"SE-ResNeXt-50 FLOPs: {flops}")