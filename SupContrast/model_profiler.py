import torch
from torch.profiler import profile, record_function, ProfilerActivity
from networks.resnet_big import SupConResNet
from torchvision.models import resnet50
import timm
# Extract the "Self CUDA time total" value
import re
import time

# Load the models
model1 = SupConResNet(name='resnet50timm').cuda()
model1.eval()

# Load the model
model2 = SupConResNet(name='seresnext50timm').cuda()
model2.eval()

# model3 = timm.create_model('resnet50.b1k_in1k', pretrained=False, num_classes=0).cuda()
# model3.eval()

# model4 = timm.create_model('seresnext50_32x4d.racm_in1k', pretrained=False, num_classes=0).cuda()
# model4.eval()

# Create dummy input
input_tensor = torch.randn(1, 3, 224, 224).cuda()

# Warm-up run to initialize CUDA context
with torch.no_grad():
    model1(input_tensor)

# Warm-up run to initialize CUDA context
with torch.no_grad():
    model2(input_tensor)

# with torch.no_grad():
#     model3(input_tensor)

# # Warm-up run to initialize CUDA context
# with torch.no_grad():
#     model4(input_tensor)

# Initialize lists to store Self CUDA time total
self_cuda_times_resnet50 = []
self_cuda_times_seresnext50 = []

# Execute the code 10 times
for i in range(30):
    print(f"Iteration {i+1}")

    # Profile the first model
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                model1(input_tensor)

    # Print the profiling results
    print(f"ResNet-50")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Extract and store the Self CUDA time total for ResNet-50
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50)
    match = re.search(r"Self CUDA time total:\s+(\d+\.\d+)ms", table)
    if match:
        self_cuda_time_total = float(match.group(1))
        print(f"Self CUDA time total: {self_cuda_time_total:.3f}ms")
    else:
        print("Self CUDA time total not found in the output.")
    self_cuda_times_resnet50.append(self_cuda_time_total)

    # Profile the second model
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                model2(input_tensor)

    # Print the profiling results
    print(f"SE-ResNeXt-50")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Extract and store the Self CUDA time total for SE-ResNeXt-50
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50)
    match = re.search(r"Self CUDA time total:\s+(\d+\.\d+)ms", table)
    if match:
        self_cuda_time_total = float(match.group(1))
        print(f"Self CUDA time total: {self_cuda_time_total:.3f}ms")
    else:
        print("Self CUDA time total not found in the output.")
    self_cuda_times_seresnext50.append(self_cuda_time_total)


# Calculate and print the average Self CUDA time total for both models in milliseconds
avg_self_cuda_time_resnet50 = sum(self_cuda_times_resnet50) / len(self_cuda_times_resnet50)
avg_self_cuda_time_seresnext50 = sum(self_cuda_times_seresnext50) / len(self_cuda_times_seresnext50)

print(f"Average Self CUDA time total for ResNet-50: {avg_self_cuda_time_resnet50:.2f} ms")
print(f"Average Self CUDA time total for SE-ResNeXt-50: {avg_self_cuda_time_seresnext50:.2f} ms")