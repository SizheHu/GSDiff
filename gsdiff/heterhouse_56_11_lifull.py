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
        self.edge_norm = nn.InstanceNorm1d(d_model)
        self.edge_global_attn = MultiHeadAttention(4, d_model)
        self.edge_feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, edges, global_attn_matrix):
        '''v-v attn.'''
        edges_normed1 = self.edge_norm(edges)



        # import cv2
        # obtain edge_global_attn_matrix
        global_attn_matrix_flatten = global_attn_matrix.reshape(edges.shape[0], -1)
        # cv2.imwrite('test.png', (global_attn_matrix_flatten.reshape(1, 2809) * 255).cpu().numpy())
        global_mat_columns = global_attn_matrix_flatten[:, :, None].repeat(1, 1, edges.shape[1])
        # cv2.imwrite('test2.png', (global_mat_columns.reshape(2809, 2809) * 255).cpu().numpy())
        global_mat_rows = global_attn_matrix_flatten[:, None, :].repeat(1, edges.shape[1], 1)
        # cv2.imwrite('test3.png', (global_mat_rows.reshape(2809, 2809) * 255).cpu().numpy())
        edges_global_matrix = torch.logical_and(global_mat_columns, global_mat_rows)
        # cv2.imwrite('test4.png', (edges_global_matrix.reshape(2809, 2809) * 255).cpu().numpy())



        # obtain edge_adjacency_attn_matrix (logical_and with global)



        # obtain edge_adjacency_attn_matrix (logical_and with global) (adaptive sample)

        # obtain edge_adjacency_attn_matrix (logical_and with global) (random sample)







        edges_attn_matrix = edges_global_matrix

        global_attn = self.edge_global_attn(edges_normed1, edges_normed1, edges_normed1, edges_attn_matrix)
        edges = edges + global_attn
        edges_normed2 = self.edge_norm(edges)
        edges = edges + self.edge_feedforward(edges_normed2)

        return edges


class EdgeModel(nn.Module):
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
        )

        self.semantics_embedding = nn.Linear(13, self.d_model // 2) # 7 == semantics.shape[2]

        self.edges_MLP = nn.Linear(self.d_model, 2)


    def forward(self, corners, global_attn_matrix, corners_padding_mask, semantics):
        '''Edge transformer'''

        '''edges_embedding (bs, 2809, 256) include padding, padding should be 0'''
        edge_coords_padding_mask = global_attn_matrix.reshape(corners.shape[0], -1, 1)
        edge_coords1 = ((corners[:, :, None, :].repeat(1, 1, corners.shape[1], 1)
                         .reshape(corners.shape[0], -1, corners.shape[2]) * 128 + 128) *
                        edge_coords_padding_mask).float()
        edge_coords2 = ((corners[:, None, :, :].repeat(1, corners.shape[1], 1, 1)
                         .reshape(corners.shape[0], -1, corners.shape[2]) * 128 + 128) *
                        edge_coords_padding_mask).float()

        edge_semans1 = ((semantics[:, :, None, :].repeat(1, 1, semantics.shape[1], 1)
                         .reshape(semantics.shape[0], -1, semantics.shape[2])) *
                        edge_coords_padding_mask).float()
        edge_semans2 = ((semantics[:, None, :, :].repeat(1, semantics.shape[1], 1, 1)
                         .reshape(semantics.shape[0], -1, semantics.shape[2])) *
                        edge_coords_padding_mask).float()

        # w in sinusoidal encoding
        div_term = (1 / 10000) ** (torch.arange(0, self.d_model // 4, 2, device=edge_coords1.device).float() / (self.d_model // 4))

        edge_coords1_embedding_sin_x = (edge_coords1[:, :, 0:1] * div_term[None, None, :].repeat(edge_coords1.shape[0],
                                                                                                 edge_coords1.shape[1],
                                                                                                 1)).sin()
        edge_coords1_embedding_cos_x = (edge_coords1[:, :, 0:1] * div_term[None, None, :].repeat(edge_coords1.shape[0],
                                                                                                 edge_coords1.shape[1],
                                                                                                 1)).cos()
        edge_coords1_embedding_x = torch.stack((edge_coords1_embedding_sin_x, edge_coords1_embedding_cos_x),
                                               dim=3).flatten(2, 3)
        edge_coords1_embedding_sin_y = (edge_coords1[:, :, 1:2] * div_term[None, None, :].repeat(edge_coords1.shape[0],
                                                                                                 edge_coords1.shape[1],
                                                                                                 1)).sin()
        edge_coords1_embedding_cos_y = (edge_coords1[:, :, 1:2] * div_term[None, None, :].repeat(edge_coords1.shape[0],
                                                                                                 edge_coords1.shape[1],
                                                                                                 1)).cos()
        edge_coords1_embedding_y = torch.stack((edge_coords1_embedding_sin_y, edge_coords1_embedding_cos_y),
                                               dim=3).flatten(2, 3)
        edge_coords1_embedding = torch.cat((edge_coords1_embedding_x, edge_coords1_embedding_y),
                                               dim=2)


        edge_coords1_embedding = torch.cat((edge_coords1_embedding, self.semantics_embedding(edge_semans1)), dim=2)

 

        edge_coords2_embedding_sin_x = (edge_coords2[:, :, 0:1] * div_term[None, None, :].repeat(edge_coords2.shape[0],
                                                                                                 edge_coords2.shape[1],
                                                                                                 1)).sin()
        edge_coords2_embedding_cos_x = (edge_coords2[:, :, 0:1] * div_term[None, None, :].repeat(edge_coords2.shape[0],
                                                                                                 edge_coords2.shape[1],
                                                                                                 1)).cos()
        edge_coords2_embedding_x = torch.stack((edge_coords2_embedding_sin_x, edge_coords2_embedding_cos_x),
                                               dim=3).flatten(2, 3)
        edge_coords2_embedding_sin_y = (edge_coords2[:, :, 1:2] * div_term[None, None, :].repeat(edge_coords2.shape[0],
                                                                                                 edge_coords2.shape[1],
                                                                                                 1)).sin()
        edge_coords2_embedding_cos_y = (edge_coords2[:, :, 1:2] * div_term[None, None, :].repeat(edge_coords2.shape[0],
                                                                                                 edge_coords2.shape[1],
                                                                                                 1)).cos()
        edge_coords2_embedding_y = torch.stack((edge_coords2_embedding_sin_y, edge_coords2_embedding_cos_y),
                                               dim=3).flatten(2, 3)
        edge_coords2_embedding = torch.cat((edge_coords2_embedding_x, edge_coords2_embedding_y),
                                             dim=2)
        edge_coords2_embedding = torch.cat((edge_coords2_embedding, self.semantics_embedding(edge_semans2)), dim=2)


        edges_embedding = edge_coords1_embedding + edge_coords2_embedding

        '''edges_center cannot be served as reference points, because no euclidean image.
           but essentially, deformable attn is [sparse] and [adaptive]. 
           so using GAT (sparse) & nn.Linear (adaptive)'''

        '''filter network, 2809 too large, set max candidates to 
           this filter can only filter false edges, we don't process padding edge here
           use deformable attention to avoid using (bs, 2809, 2809) attention mask'''

        for layer in self.transformer_layers:
            edges_embedding = layer(edges_embedding, global_attn_matrix)

        output_edges = self.edges_MLP(edges_embedding)

        return output_edges
