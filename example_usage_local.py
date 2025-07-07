import torch
from open_clip import create_model_and_transforms, tokenize
from PIL import Image

# Load model
model_path = "src/logs_test/2025_07_06-23_16_02-model_ViT-B-32-lr_1e-05-b_2-j_0-p_amp/checkpoints/epoch_1.pt"
model, _, preprocess = create_model_and_transforms(
    'ViT-B-32', pretrained=model_path, device='cuda:1'  # or 'cpu'
)
model.eval()

# Prepare example input
image = preprocess(Image.open("mydataset/images/0003.png")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.autocast("cuda:1"):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]




