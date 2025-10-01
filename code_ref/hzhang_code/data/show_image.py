import torch

# Path to your tensor
tensor_path = "00053d900f1caf6c6eba950ff36c17025fa814e48cf36de126cb41382a5063b2.pt"

# Load
tensor = torch.load(tensor_path)

# Show details
print(type(tensor))
if isinstance(tensor, torch.Tensor):
    print("Tensor shape:", tensor.shape)
else:
    # Sometimes torch.save can store dicts or other objects
    try:
        for k, v in tensor.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: shape {v.shape}")
            else:
                print(f"{k}: type {type(v)}")
    except AttributeError:
        print("Loaded object is not a Tensor or dict. Type:", type(tensor))

import matplotlib.pyplot as plt
# Loop through N slices
tensor_np = tensor.cpu().numpy()
for i in range(tensor_np.shape[0]):
    img = tensor_np[i]  # shape [3, 224, 224]
    img = img.transpose(1, 2, 0)  # to [224, 224, 3] for matplotlib

    plt.imshow(img)
    plt.title(f"Slice {i+1}/{tensor_np.shape[0]}")
    plt.axis("off")
    plt.show()