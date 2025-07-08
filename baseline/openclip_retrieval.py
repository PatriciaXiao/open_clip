import os
import random
import torch
import webdataset as wds
import numpy as np
from tqdm import tqdm
from torch.nn.functional import cosine_similarity

from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image

# Set up device
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Load the model and config files from the Hugging Face Hub
print("Loading model and tokenizer")
model_name = "ViT-L-16"
pretrained = "openai"  # or "laion2b_s32b_b82k" if you prefer LAION pretraining
#model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
#tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
print("Loading OpenCLIP ViT-L/16 model and tokenizer")
model, _, preprocess = create_model_from_pretrained(model_name, pretrained=pretrained, device=device)
tokenizer = get_tokenizer(model_name)
model = model.to(device)
print("\tFinished loading")

# Parameters
#webdataset_path = "/projects/chimeranb/patxiao/mydata.tar"  # <- CHANGE THIS
#sample_fraction = 0.2
#max_sample_size = 5000

webdataset_path = "../mydataset/sample_data/my_sample.tar"
sample_fraction = 1.0
max_sample_size = 5000

# Load and decode dataset
print(f"Loading data from {webdataset_path}")
dataset = (
    wds.WebDataset(webdataset_path)
    .decode("pil")  # Decode image bytes to PIL images
    .to_tuple("png", "txt")  # Unpack image/text pair
    .map(lambda sample: (sample[0].convert("RGB"), sample[1]))  # Convert image to RGB safely
    .slice(max_sample_size)  # Select up to 5000 samples
)
print("\tFinished loading")

# Buffer entire dataset and sample
buffered_data = list(iter(dataset))
sample_size = int(len(buffered_data) * sample_fraction)
sampled_data = random.sample(buffered_data, sample_size)

# Encode all images and texts
image_embeddings = []
text_embeddings = []
model.eval()

print("Run samples")
with torch.no_grad():
    for img, txt in tqdm(sampled_data, desc="Encoding"):
        image_tensor = preprocess(img).unsqueeze(0).to(device)
        image_emb = model.encode_image(image_tensor)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        image_embeddings.append(image_emb.cpu())

        text_tokens = tokenizer([txt]).to(device)
        text_emb = model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        text_embeddings.append(text_emb.cpu())

# Stack embeddings
image_embeddings = torch.cat(image_embeddings, dim=0)
text_embeddings = torch.cat(text_embeddings, dim=0)

# Compute cosine similarity
similarity = cosine_similarity(text_embeddings.unsqueeze(1), image_embeddings.unsqueeze(0), dim=-1)

# Compute ranks
ranks = []
for i in range(len(sampled_data)):
    sim_scores = similarity[i]
    rank = torch.argsort(sim_scores, descending=True).tolist().index(i) + 1
    ranks.append(rank)

# Evaluate
ranks = np.array(ranks)
print("\n\U0001F4CA Retrieval Evaluation Results")
print(f"Recall@1:  {np.mean(ranks <= 1):.4f}")
print(f"Recall@5:  {np.mean(ranks <= 5):.4f}")
print(f"Recall@10: {np.mean(ranks <= 10):.4f}")
print(f"Mean Rank: {np.mean(ranks):.2f}")
print(f"Median Rank: {np.median(ranks):.2f}")
