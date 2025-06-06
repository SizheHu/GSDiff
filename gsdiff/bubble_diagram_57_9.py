import math
import torch
import torch.nn as nn
import torch.nn.functional as F

'''This is the Transformer used for topology encoding and decoding'''

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_subspace = d_model // heads
        self.WQ = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.fusion = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        batch_size = k.shape[0]
        # (batch_size, heads, seq_length, repr_dim // heads)
        k = self.WK(k).reshape(batch_size, -1, self.heads, self.d_subspace).transpose(1, 2)
        q = self.WQ(q).reshape(batch_size, -1, self.heads, self.d_subspace).transpose(1, 2)
        v = self.WV(v).reshape(batch_size, -1, self.heads, self.d_subspace).transpose(1, 2)
        # (batch_size, 1, seq_length, seq_length)
        mask = mask[:, None, :, :]
        # calculate attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_subspace)
        scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = torch.matmul(scores, v)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        output = self.fusion(concat)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.semantics_norm = nn.InstanceNorm1d(d_model)
        self.attn = MultiHeadAttention(4, d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, semantics, attn_matrix):
        '''v-v attn.'''
        semantics_normed1 = self.semantics_norm(semantics)
        attn = self.attn(semantics_normed1, semantics_normed1, semantics_normed1, attn_matrix)
        semantics = semantics + attn
        semantics_normed2 = self.semantics_norm(semantics)
        semantics = semantics + self.feedforward(semantics_normed2)

        return semantics


class TopoGraphModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 256

        self.transformer_layers = nn.Sequential(
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model)
        )

        self.semantics_embedding = nn.Linear(7, self.d_model) # 7 == semantics.shape[2]
        self.semantics_MLP = nn.Linear(self.d_model, 7)  # 7 == semantics.shape[2]
        self.edges_MLP = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 2),
        )


    def forward(self, semantics, adjacency_matrix, semantics_padding_mask, global_matrix):
        semantics = semantics.float()
        semantics_embedding = self.semantics_embedding(semantics)

        for layer in self.transformer_layers:
            semantics_embedding = layer(semantics_embedding, adjacency_matrix)

        edges_padding_mask = global_matrix.reshape(global_matrix.shape[0], -1, 1)

        edge_semantics1 = semantics_embedding[:, :, None, :].repeat(1, 1, semantics_embedding.shape[1], 1).reshape(semantics_embedding.shape[0], -1, semantics_embedding.shape[2])
        edge_semantics2 = semantics_embedding[:, None, :, :].repeat(1, semantics_embedding.shape[1], 1, 1).reshape(semantics_embedding.shape[0], -1, semantics_embedding.shape[2])

        edge_semantics = edge_semantics1 + edge_semantics2

        output_semantics = self.semantics_MLP(semantics_embedding)
        output_edges = self.edges_MLP(edge_semantics)


        return output_semantics, output_edges
