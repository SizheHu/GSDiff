import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        scores = scores.masked_fill(mask == False, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = torch.matmul(scores, v)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        output = self.fusion(concat)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.corner_norm = nn.InstanceNorm1d(d_model)
        self.corner_global_attn = MultiHeadAttention(4, d_model)
        self.corner_feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )


    def forward(self, corners, global_attn_matrix):
        '''v-v attn. use Pre-Norm.'''
        corners_normed1 = self.corner_norm(corners)
        global_attn = self.corner_global_attn(corners_normed1, corners_normed1, corners_normed1, global_attn_matrix)
        corners = corners + global_attn
        corners_normed2 = self.corner_norm(corners)
        corners = corners + self.corner_feedforward(corners_normed2)

        return corners


class HeterHouseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 256
        self.time_embedding1 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.semantics_embedding = nn.Linear(8, self.d_model)

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

        self.corners_MLP1 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 2)
        )

        self.corners_MLP2 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 7+1)
        )

    def forward(self, corners_withsemantics_t, global_attn_matrix, t):
        corners_t = corners_withsemantics_t[:, :, :2]
        semantics_t = corners_withsemantics_t[:, :, 2:]

        corners_t = (corners_t * 128 + 128).float()

        # w in sinusoidal encoding
        div_term = (1 / 10000) ** (torch.arange(0, self.d_model // 2, 2, device=corners_t.device).float() / (self.d_model // 2))

        corners_embedding_sin_x = (corners_t[:, :, 0:1] * div_term[None, None, :].repeat(corners_t.shape[0],
                                                                                                 corners_t.shape[1],
                                                                                                 1)).sin()
        corners_embedding_cos_x = (corners_t[:, :, 0:1] * div_term[None, None, :].repeat(corners_t.shape[0],
                                                                                                 corners_t.shape[1],
                                                                                                 1)).cos()
        corners_embedding_x = torch.stack((corners_embedding_sin_x, corners_embedding_cos_x),
                                               dim=3).flatten(2, 3)
        corners_embedding_sin_y = (corners_t[:, :, 1:2] * div_term[None, None, :].repeat(corners_t.shape[0],
                                                                                                 corners_t.shape[1],
                                                                                                 1)).sin()
        corners_embedding_cos_y = (corners_t[:, :, 1:2] * div_term[None, None, :].repeat(corners_t.shape[0],
                                                                                                 corners_t.shape[1],
                                                                                                 1)).cos()
        corners_embedding_y = torch.stack((corners_embedding_sin_y, corners_embedding_cos_y),
                                               dim=3).flatten(2, 3)
        corners_embedding = torch.cat((corners_embedding_x, corners_embedding_y),
                                           dim=2)

        '''semantics embedding'''
        semantics_embedding = self.semantics_embedding(semantics_t.float())

        t_freqs = torch.exp(-math.log(10000) *
                            torch.arange(start=0, end=self.d_model // 2, dtype=torch.float32) /
                            (self.d_model // 2)
                            ).to(semantics_embedding.device)
        t_sinusoidal_encoding = torch.cat(
            [torch.cos(t.unsqueeze(1).float() * t_freqs.unsqueeze(0)),
             torch.sin(t.unsqueeze(1).float() * t_freqs.unsqueeze(0))]
            , dim=1)
        time_embedding_corners = self.time_embedding1(t_sinusoidal_encoding).unsqueeze(1).repeat(
            (1, corners_t.shape[1], 1))

        corners_total_embedding = corners_embedding + semantics_embedding + time_embedding_corners

        for heterogeneous_layer in self.transformer_layers:
            corners_total_embedding = heterogeneous_layer(corners_total_embedding,
                                                          global_attn_matrix)

        output_corners1 = self.corners_MLP1(corners_total_embedding)
        output_corners2 = self.corners_MLP2(corners_total_embedding)

        return output_corners1, output_corners2
