import torch
import torch.nn as nn

class LoRAAttention(nn.Module):
    def __init__(self, embed_dim, r):
        super(LoRAAttention, self).__init__()
        self.embed_dim = embed_dim
        self.r = r

        self.Q_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.Q_weight.requires_grad = False 
        self.K_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.K_weight.requires_grad = False 
        self.V_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.V_weight.requires_grad = False 

        self.Q_A = nn.Parameter(torch.empty(r, embed_dim))
        self.Q_B = nn.Parameter(torch.zeros(embed_dim, r))
        self.K_A = nn.Parameter(torch.empty(r, embed_dim))
        self.K_B = nn.Parameter(torch.zeros(embed_dim, r))
        self.V_A = nn.Parameter(torch.empty(r, embed_dim))
        self.V_B = nn.Parameter(torch.zeros(embed_dim, r))

        nn.init.normal_(self.Q_A, mean=0.0, std=0.02)
        nn.init.normal_(self.K_A, mean=0.0, std=0.02)
        nn.init.normal_(self.V_A, mean=0.0, std=0.02)

        self.O_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, query, key, value):
        # query, key, value shape: (batch_size, seq_len, embed_dim)
        Q_orginal = nn.functional.linear(query, self.Q_weight) # shape: (batch_size, seq_len, embed_dim)
        K_orginal = nn.functional.linear(key, self.K_weight) 
        V_orginal = nn.functional.linear(value, self.V_weight)

        Q_delta_A = torch.matmul(query, self.Q_A.t())  # shape: (batch_size, seq_len, r)
        Q_delta = torch.matmul(Q_delta_A, self.Q_B.t())  # shape: (batch_size, seq_len, embed_dim)
        K_delta_A = torch.matmul(key, self.K_A.t())  # shape: (batch_size, seq_len, r)
        K_delta = torch.matmul(K_delta_A, self.K_B.t())  # shape: (batch_size, seq_len, embed_dim)
        V_delta_A = torch.matmul(value, self.V_A.t())  # shape: (batch_size, seq_len, r)
        V_delta = torch.matmul(V_delta_A, self.V_B.t())  # shape: (batch_size, seq_len, embed_dim)

        Q_current = Q_orginal + Q_delta # shape: (batch_size, seq_len, embed_dim)
        K_current = K_orginal + K_delta
        V_current = V_orginal + V_delta

        QV_mapping = torch.matmul(Q_current, K_current.transpose(-2, -1)) / (self.embed_dim**0.5) # shape: (batch_size, seq_len, seq_len)
        attention_scores = torch.softmax(QV_mapping, dim=-1) # shape: (batch_size, seq_len, seq_len)
        output = torch.matmul(attention_scores, V_current) # shape: (batch_size, seq_len, embed_dim)
        output = nn.functional.linear(output, self.O_weight) # shape: (batch_size, seq_len, embed_dim)

        return output