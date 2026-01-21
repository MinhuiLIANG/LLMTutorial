import attention_lora
import torch

embed_dim = 128
r = 8
batch_size = 4
seq_len = 16
torch.manual_seed(42)

model = attention_lora.LoRAAttention(embed_dim, r)
query = torch.randn(batch_size, seq_len, embed_dim)
key = torch.randn(batch_size, seq_len, embed_dim)   
value = torch.randn(batch_size, seq_len, embed_dim)

output = model(query, key, value)
print("output: ", output)
print("output shape: ", output.shape)