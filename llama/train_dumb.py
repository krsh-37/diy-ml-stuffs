import torch
from tqdm import tqdm

from dataclasses import dataclass
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from model import LLAMAModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
@dataclass
class ModelConfig:
    d_model = 512
    seq_len = 64
    batch_size = 2
    n_blocks = 6
    q_heads = 8
    kv_heads = 4
    ffn_multiplier = None
    ffn_muliple_of = 32
    vocab_size = 64 # set later
    eps = 1e-6

def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    model.eval()
    tokens = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
    tokens = model.generate(tokens, max_new_tokens)
    model.train()

    return tokenizer.decode(tokens[0], skip_special_tokens=True)

data_path = "../data/hp_book1.txt"
def pre_process_text(all_files):
    all_words = ""
    for f_name in all_files:
        all_lines = open(f_name).readlines()
        all_lines = [ l.replace("\n", "")
                     for l in all_lines if not 'Page | ' in l]
        text = "".join(all_lines).lower()
        all_words+=text
    return all_words

txt = pre_process_text([data_path])

print("Loading Llama 2 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token # Set padding token for batching

# Update vocab_size in config
config = ModelConfig()
config.vocab_size = len(tokenizer)

model = LLAMAModel(config)

print("Tokenizing text...")
tokens = tokenizer.encode(txt, return_tensors='pt')[0]

# prepare data
inputs = []
targets = []
for i in range(len(tokens) - config.seq_len):
    inputs.append(tokens[i:i+config.seq_len])
    targets.append(tokens[i+1:i+config.seq_len+1])

inputs = torch.stack(inputs)
targets = torch.stack(targets)

dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

# Model, Optimizer, and Training
model = LLAMAModel(config).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print(f"Starting training on {DEVICE}...")
num_epochs = 3 

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    i = 0
    for batch_inputs, batch_targets in tqdm(dataloader):
        i += 1
        batch_inputs, batch_targets = batch_inputs.to(DEVICE), batch_targets.to(DEVICE)

        optimizer.zero_grad()
        _, loss = model(batch_inputs, targets = batch_targets)
        
        if loss is not None:
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (i + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
            print( generate_text(model, tokenizer, "<|begin_of_text|>", max_new_tokens=100) )

    print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")

# Inference
print("\n--- Generating Text ---")
prompt = "The magic of"
generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=100)
print(f"Prompt: '{prompt}'")
print(f"Generated Text: {generated_text}")