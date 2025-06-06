import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


'''This is the boundary-constrained edge prediction Transformer with the random self-supervision strategy.'''


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, x):
        not_mask = ~(torch.zeros(x.shape[0], x.shape[2], x.shape[3]).to(x.device).to(torch.bool)) # False
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * (2 * math.pi)
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * (2 * math.pi)
        dim_t = torch.arange(128, dtype=torch.float32, device=x.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / 128)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


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
        self.cross_attn = MultiHeadAttention(4, d_model)
        self.edge_feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, edges, global_attn_matrix, x, cross_attn_mask):
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


        '''单方面从房间中汇聚信息。'''
        edges_normed2 = self.edge_norm(edges)
        cross_attn = self.cross_attn(edges_normed2, x, x, cross_attn_mask)
        edges = edges + cross_attn

        '''FFN'''
        edges_normed3 = self.edge_norm(edges)
        edges = edges + self.edge_feedforward(edges_normed3)

        return edges


class BoundEdgeModel(nn.Module):
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

        self.semantics_embedding = nn.Linear(7, self.d_model // 2) # 7 == semantics.shape[2]

        self.edges_MLP = nn.Linear(self.d_model, 2)

        self.lambdas_MLP = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, 1)
        )





        
        self.proj16 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=(1, 1)),
            nn.InstanceNorm2d(num_features=256, eps=1e-05, affine=True)
        )
        # self.bn16 = FrozenBatchNorm2d(1024)
        # self.activation = nn.ReLU()
        self.sinopos = PositionEmbeddingSine()
        # self.layer_embedding = nn.Parameter(torch.Tensor(1, 256)) # 我们选用了1个尺度的特征图





    

    def forward(self, corners, global_attn_matrix, corners_padding_mask, semantics, feat_16):
        batch_size = corners.shape[0]

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



        random_lambda = torch.rand((corners.shape[0], 2809, 1), device=corners.device)
        edge_coords3 = edge_coords1 * random_lambda + edge_coords2 * (1 - random_lambda)
        edge_semans3 = torch.zeros_like(edge_semans1) # 随机插值点无法得知语义



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









        edge_coords3_embedding_sin_x = (edge_coords3[:, :, 0:1] * div_term[None, None, :].repeat(edge_coords3.shape[0],
                                                                                                 edge_coords3.shape[1],
                                                                                                 1)).sin()
        edge_coords3_embedding_cos_x = (edge_coords3[:, :, 0:1] * div_term[None, None, :].repeat(edge_coords3.shape[0],
                                                                                                 edge_coords3.shape[1],
                                                                                                 1)).cos()
        edge_coords3_embedding_x = torch.stack((edge_coords3_embedding_sin_x, edge_coords3_embedding_cos_x),
                                               dim=3).flatten(2, 3)
        edge_coords3_embedding_sin_y = (edge_coords3[:, :, 1:2] * div_term[None, None, :].repeat(edge_coords3.shape[0],
                                                                                                 edge_coords3.shape[1],
                                                                                                 1)).sin()
        edge_coords3_embedding_cos_y = (edge_coords3[:, :, 1:2] * div_term[None, None, :].repeat(edge_coords3.shape[0],
                                                                                                 edge_coords3.shape[1],
                                                                                                 1)).cos()
        edge_coords3_embedding_y = torch.stack((edge_coords3_embedding_sin_y, edge_coords3_embedding_cos_y),
                                               dim=3).flatten(2, 3)
        edge_coords3_embedding = torch.cat((edge_coords3_embedding_x, edge_coords3_embedding_y),
                                           dim=2)
        edge_coords3_embedding = torch.cat((edge_coords3_embedding, self.semantics_embedding(edge_semans3)), dim=2)









        edges_embedding = edge_coords1_embedding + edge_coords2_embedding + edge_coords3_embedding








        pos_16 = self.sinopos(feat_16).permute(0, 2, 3, 1).view(batch_size, 16**2, 256) # (bs, 256, 256)
        proj_16 = self.proj16(feat_16).permute(0, 2, 3, 1).view(batch_size, 16**2, 256) # (bs, 256, 256)
        x = pos_16 + proj_16 # (bs, 256, 256)

        # 交叉注意力padding掩码 (bs, 2809, 256)
        cross_attn_mask = global_attn_matrix.view(batch_size, 2809, 1).repeat(1, 1, x.shape[1]).to(x.dtype)













        

        for layer in self.transformer_layers:
            edges_embedding = layer(edges_embedding, global_attn_matrix, x, cross_attn_mask)






        
        output_edges = self.edges_MLP(edges_embedding)

        output_lambdas = self.lambdas_MLP(edges_embedding)

        return output_edges, random_lambda, output_lambdas
