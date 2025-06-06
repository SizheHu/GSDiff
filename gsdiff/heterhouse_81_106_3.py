import math
import torch
import torch.nn as nn
import torch.nn.functional as F

'''This is the node generation transformer used for boundary constraints.'''

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
        d = 256
        dim_t = torch.arange(d // 2, dtype=torch.float32, device=x.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / (d // 2))
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        # 防止反向传播更新
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

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

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, heads, d_model, d_embed):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_subspace = d_model // heads

        self.WQ = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_embed, d_model)
        self.WK = nn.Linear(d_embed, d_model)

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
        self.cross_attn = MultiHeadCrossAttention(4, d_model, d_embed=256)
        self.corner_feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )


    def forward(self, corners, global_attn_matrix, x, cross_attn_matrix):
        '''v-v attn. use Pre-Norm.'''
        corners_normed1 = self.corner_norm(corners)
        global_attn = self.corner_global_attn(corners_normed1, corners_normed1, corners_normed1, global_attn_matrix)
        corners = corners + global_attn
        '''v-i attn (image as kv, v as q.)'''
        corners_normed2 = self.corner_norm(corners)
        # print(x.shape[2], corners_normed2.shape[2])
        cross_attn = self.cross_attn(corners_normed2, x, x, cross_attn_matrix)
        corners = corners + cross_attn
        '''ffn'''
        corners_normed3 = self.corner_norm(corners)
        corners = corners + self.corner_feedforward(corners_normed3)

        return corners


class BoundHeterHouseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 512
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










        # self.proj64 = nn.InstanceNorm2d(num_features=256, eps=1e-05, affine=True)
        # self.proj32 = nn.Sequential(
        #     nn.Conv2d(512, 256, kernel_size=(1, 1)),
        #     nn.InstanceNorm2d(num_features=256, eps=1e-05, affine=True)
        # )
        # self.proj16 = nn.Sequential(
        #     nn.Conv2d(1024, 256, kernel_size=(1, 1)),
        #     nn.InstanceNorm2d(num_features=256, eps=1e-05, affine=True)
        # )
        # self.sinopos = PositionEmbeddingSine()
        # self.layer_embedding = nn.Parameter(torch.Tensor(3, 256)) # 我们选用了3个尺度的特征图


        # self.proj32 = nn.Sequential(
        #     nn.Conv2d(512, 256, kernel_size=(1, 1)),
        #     nn.InstanceNorm2d(num_features=256, eps=1e-05, affine=True)
        # )
        # self.proj16 = nn.Sequential(
        #     nn.Conv2d(1024, 256, kernel_size=(1, 1)),
        #     nn.InstanceNorm2d(num_features=256, eps=1e-05, affine=True)
        # )
        # self.bn32 = FrozenBatchNorm2d(512)
        # self.bn16 = FrozenBatchNorm2d(1024)
        # self.activation = nn.ReLU()
        # self.sinopos = PositionEmbeddingSine()
        # self.layer_embedding = nn.Parameter(torch.Tensor(2, 256)) # 我们选用了2个尺度的特征图


        self.proj16 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=(1, 1)),
            nn.InstanceNorm2d(num_features=256, eps=1e-05, affine=True)
        )
        # self.bn16 = FrozenBatchNorm2d(1024)
        # self.activation = nn.ReLU()
        self.sinopos = PositionEmbeddingSine()
        # self.layer_embedding = nn.Parameter(torch.Tensor(1, 256)) # 我们选用了1个尺度的特征图



        


        





    

    # def forward(self, corners_withsemantics_t, global_attn_matrix, t, feat_64, feat_32, feat_16):
    # def forward(self, corners_withsemantics_t, global_attn_matrix, t, feat_32, feat_16):
    def forward(self, corners_withsemantics_t, global_attn_matrix, t, feat_16):
        batch_size = corners_withsemantics_t.shape[0]



        
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











        # pos_64 = self.sinopos(feat_64).permute(0, 2, 3, 1).view(batch_size, 64**2, 256) # (bs, 4096, 256)
        # pos_32 = self.sinopos(feat_32).permute(0, 2, 3, 1).view(batch_size, 32**2, 256) # (bs, 1024, 256)
        # pos_16 = self.sinopos(feat_16).permute(0, 2, 3, 1).view(batch_size, 16**2, 256) # (bs, 256, 256)
        # proj_64 = self.proj64(feat_64).permute(0, 2, 3, 1).view(batch_size, 64**2, 256) # (bs, 4096, 256)
        # proj_32 = self.proj32(feat_32).permute(0, 2, 3, 1).view(batch_size, 32**2, 256) # (bs, 1024, 256)
        # proj_16 = self.proj16(feat_16).permute(0, 2, 3, 1).view(batch_size, 16**2, 256) # (bs, 256, 256)

        # x_64 = pos_64 + proj_64 + self.layer_embedding[0].view(1, 1, self.d_model) # (bs, 4096, 256)
        # x_32 = pos_32 + proj_32 + self.layer_embedding[1].view(1, 1, self.d_model) # (bs, 1024, 256)
        # x_16 = pos_16 + proj_16 + self.layer_embedding[2].view(1, 1, self.d_model) # (bs, 256, 256)

        # x = torch.cat((x_64, x_32, x_16), dim=1) # (bs, 5376, 256)





        # feat_32 = self.activation(self.bn32(feat_32))
        # feat_16 = self.activation(self.bn16(feat_16))
        # pos_32 = self.sinopos(feat_32).permute(0, 2, 3, 1).view(batch_size, 32**2, 256) # (bs, 1024, 256)
        # pos_16 = self.sinopos(feat_16).permute(0, 2, 3, 1).view(batch_size, 16**2, 256) # (bs, 256, 256)
        # proj_32 = self.proj32(feat_32).permute(0, 2, 3, 1).view(batch_size, 32**2, 256) # (bs, 1024, 256)
        # proj_16 = self.proj16(feat_16).permute(0, 2, 3, 1).view(batch_size, 16**2, 256) # (bs, 256, 256)

        # x_32 = pos_32 + proj_32 + self.layer_embedding[0].view(1, 1, self.d_model) # (bs, 1024, 256)
        # x_16 = pos_16 + proj_16 + self.layer_embedding[1].view(1, 1, self.d_model) # (bs, 256, 256)

        # x = torch.cat((x_32, x_16), dim=1) # (bs, 5376, 256)




        # feat_16 = self.activation(self.bn16(feat_16))
        # print(feat_16.max(), feat_16.min(), feat_16.mean(), feat_16.sum())
        # assert 0
        pos_16 = self.sinopos(feat_16).permute(0, 2, 3, 1).view(batch_size, 16**2, 256) # (bs, 256, 256)
        proj_16 = self.proj16(feat_16).permute(0, 2, 3, 1).view(batch_size, 16**2, 256) # (bs, 256, 256)
        x = pos_16 + proj_16 # (bs, 256, 256)

        
        




        # 交叉注意力padding掩码 (bs, 53, 5376)
        cross_attn_matrix = global_attn_matrix[:, :, 0:1].repeat(1, 1, x.shape[1]).to(x.dtype)
        










        

        for heterogeneous_layer in self.transformer_layers:
            corners_total_embedding = heterogeneous_layer(corners_total_embedding,
                                                          global_attn_matrix, x, cross_attn_matrix)

        output_corners1 = self.corners_MLP1(corners_total_embedding)
        output_corners2 = self.corners_MLP2(corners_total_embedding)

        return output_corners1, output_corners2
