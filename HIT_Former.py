from logging import getLogger

import math
import numpy
import numpy as np
import torch
import torch.nn as nn

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, BatchNorm
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp

from libcity.utils.positional_encodings import get_pos_encoder

class RGCN_batch(nn.Module):
    """
    一个层数可调，且能以向量化方式高效处理批量时序图数据 [B, L, N, C] 的RGCN模型。
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_relations: int, num_layers: int, relation_mx: np.ndarray, dropout: float = 0.5):
        # 修正了super()中的类名
        super(RGCN_batch, self).__init__()

        # --- 预处理邻接关系，这部分不变 ---
        E1 = relation_mx.copy()
        edge_index, edge_attr_values = from_scipy_sparse_matrix(sp.csr_matrix(E1))
        # 将 edge_index 和 edge_type 注册为模型的缓冲区(buffer)，这样它们会自动随模型移动到CPU/GPU
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_type', edge_attr_values.long() - 1)

        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1.")

        self.num_layers = num_layers
        self.dropout = dropout

        # --- GCN层的定义保持不变，因为它们作用于单个图的特征维度 ---
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_channels
            if i == 0:
                in_dim = in_channels

            self.convs.append(RGCNConv(in_dim, hidden_channels, num_relations))

            if i < num_layers - 1:
                # 修正了BatchNorm的类名，并使其作用于隐藏层通道数
                self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.final_fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 (高效的向量化版本)

        Args:
            x (torch.Tensor): 节点特征矩阵，形状为 [B, L, N, C]

        Returns:
            torch.Tensor: 模型输出，形状为 [B, L, N, out_channels]
        """
        batch_size, seq_len, num_nodes, in_channels = x.shape
        num_snapshots = batch_size * seq_len
        device = x.device

        # --- 1. 节点特征压平 ---
        # 将输入从 [B, L, N, C] 重塑为 [B*L*N, C]
        # 现在我们有了一个包含所有节点在所有时间步的特征的长列表
        x_flat = x.view(num_snapshots * num_nodes, in_channels)

        # --- 2. 邻接关系扩展 ---
        # 创建一个应用于“超级图”的 batched_edge_index
        # 首先复制原始的 edge_index B*L 次
        batched_edge_index = self.edge_index.repeat(1, num_snapshots)

        # 然后为每个快照的边索引加上偏移量
        # 偏移量形如 [0, 0, ..., N, N, ..., 2N, 2N, ...]
        offset = torch.arange(num_snapshots, device=device).view(-1, 1).repeat(1, self.edge_index.size(1))
        offset = offset.view(-1) * num_nodes
        batched_edge_index += offset

        # 边的类型也需要相应地复制
        batched_edge_type = self.edge_type.repeat(num_snapshots)

        # --- 3. GCN 计算 (在“超级图”上进行一次计算) ---
        current_x = x_flat
        for layer_idx in range(self.num_layers):
            # 将压平后的节点特征和扩展后的邻接关系传入GCN层
            current_x = self.convs[layer_idx](current_x, batched_edge_index, batched_edge_type)
            if layer_idx < self.num_layers - 1:
                # BatchNorm1d期望的输入是 [N, C]，我们的输入是 [B*L*N, C]，完全匹配
                current_x = self.bns[layer_idx](current_x)
                current_x = F.relu(current_x)
                current_x = F.dropout(current_x, p=self.dropout, training=self.training)

        # GCN处理后, current_x 的形状是 [B*L*N, hidden_channels]

        # --- 4. 恢复形状 ---
        # 首先，应用最终的全连接层，它作用于最后一个维度
        output = self.final_fc(current_x)  # 输出形状: [B*L*N, out_channels]

        # 最后，将形状恢复为 [B, L, N, out_channels]
        output = output.view(batch_size, seq_len, num_nodes, -1)

        return output

class IntersectionFusionLayer(nn.Module):
    """
    将交叉口节点(X)的特征融合回转向节点(T)，并恢复原始图形状。
    """

    def __init__(self, num_R, num_T, num_X, intersection_groups):
        super().__init__()
        self.num_R = num_R
        self.num_T = num_T
        self.num_X = num_X
        self.num_original_nodes = num_R + num_T
        self.intersection_groups = intersection_groups
        # 可以选择更复杂的融合方式，例如一个线性层
        # self.fusion_linear = nn.Linear(D_out * 2, D_out)

    def forward(self, x_extended):
        """
        Args:
            x_extended (torch.Tensor): RGCN在扩展图上的输出, [B, L, N_total, D_out]

        Returns:
            torch.Tensor: 融合后的特征，形状为 [B, L, N_original, D_out]
        """
        # 1. 先分离出原始节点(R, T)和交叉口节点(X)的特征
        x_original = x_extended[:, :, :self.num_original_nodes, :]
        x_intersections = x_extended[:, :, self.num_original_nodes:, :]

        # 创建一个副本用于更新，避免原地修改
        x_fused = x_original.clone()

        # 2. 遍历每个交叉口，将其特征加到对应的转向节点上
        for i, group in enumerate(self.intersection_groups):
            # 获取当前交叉口X_i的特征
            # x_intersection_i shape: [B, L, 1, D_out]
            x_intersection_i = x_intersections[:, :, i:i + 1, :]

            # 提取该交叉口所有T节点的当前特征
            # group_indices = torch.tensor(group, device=x_extended.device).long()
            # current_t_features = x_fused[:, :, group_indices, :]

            # 执行融合操作 (这里使用简单的相加)
            # x_intersection_i会通过广播自动扩展
            fused_t_features = x_fused[:, :, group, :] + x_intersection_i

            # 将更新后的T节点特征放回原位
            x_fused[:, :, group, :] = fused_t_features

            # 更复杂的融合方式示例:
            # t_and_x_concat = torch.cat([current_t_features, x_intersection_i.expand(-1, -1, len(group), -1)], dim=-1)
            # fused_t_features = self.fusion_linear(t_and_x_concat)
            # x_fused[:, :, group, :] = fused_t_features

        return x_fused

class HierarchicalRGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_layers,
                 extended_mx, num_R, num_T, num_X, intersection_groups):
        super().__init__()
        print("使用RGCN_batch")
        self.rgcn_extended = RGCN_batch(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,  # RGCN的输出维度
            num_relations=num_relations,
            num_layers=num_layers,
            relation_mx=extended_mx
        )
        self.fusion_layer = IntersectionFusionLayer(
            num_R, num_T, num_X, intersection_groups
        )

    def forward(self, x_extended):
        """
        Args:
            x_extended (torch.Tensor): 扩展图的输入特征, [B, L, N_total, C_in]
        """
        # 1. 在扩展图上进行信息传播
        hidden_extended = self.rgcn_extended(x_extended)

        # 2. 融合交叉口信息并恢复原始形状
        output_original = self.fusion_layer(hidden_extended)

        return output_original

class GraphBiasModule(nn.Module):
    """
    根据外部邻接矩阵 E2 (距离) 和 E3 (流量) 计算注意力偏置项 b。
    b = fusion(fc1(norm1(E2)) + fc2(norm2(E3)))
    """

    def __init__(self, hidden_dim, sigma=1.0):
        """
        Args:
            hidden_dim (int): fc层的输出维度，应与注意力头的数量或维度相关。
            sigma (float): 用于距离归一化的高斯核函数的sigma值。
        """
        super().__init__()
        # 两个独立的线性层
        self.fc1 = nn.Linear(1, hidden_dim, bias=False)
        self.fc2 = nn.Linear(1, hidden_dim, bias=False)
        self.sigma_squared = sigma ** 2

        # 简单的融合层，这里使用ReLU作为激活函数
        self.fusion = nn.ReLU()

    def _norm_distance(self, E2):
        """
        归一化距离矩阵E2。
        使用高斯核函数 exp(-d^2 / sigma^2)。
        无穷大的距离会被映射到0。
        """
        # 将无穷大替换为一个非常大的数，以保证数值稳定性
        E2 = torch.where(torch.isinf(E2), torch.full_like(E2, 1e6), E2)
        return torch.exp(-E2.pow(2) / self.sigma_squared)

    def _norm_flow(self, E3):
        """
        归一化流量矩阵E3。
        使用 log1p -> tanh。
        """
        # log1p 缓解极端值影响，tanh 缩放到 [-1, 1]
        return torch.tanh(torch.log1p(E3))

    def forward(self, trajDist_mx, flowCnt_mx):
        """
        Args:
            E2 trajDist_mx (torch.Tensor): 路径距离矩阵, shape [N, N]
            E3 flowCnt_mx  (torch.Tensor): 流量统计矩阵, shape [N, N]

        Returns:
            torch.Tensor: 计算出的偏置 b, shape [hidden_dim, N, N]
        """
        E2 = trajDist_mx.detach().clone()
        E3 = flowCnt_mx.detach().clone()
        # 1. 归一化
        norm_e2 = self._norm_distance(E2)
        norm_e3 = self._norm_flow(E3)

        # 2. 通过FC层
        # 输入需要是 [..., 1] 的形状
        b1 = self.fc1(norm_e2.unsqueeze(-1))  # [N, N, hidden_dim]
        b2 = self.fc2(norm_e3.unsqueeze(-1))  # [N, N, hidden_dim]

        # 3. 融合
        # 将hidden_dim维度换到前面，方便后续广播
        # [N, N, hidden_dim] -> [hidden_dim, N, N]
        b = self.fusion(b1 + b2).permute(2, 0, 1)

        return b

class AGTStyleGraphAttention(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_heads, config, dropout=0.1, temper=0.7):
        """
        Args:
            embed_dim (int): 输入和输出的维度 (C)
            hidden_size (int): 注意力内部隐藏维度 (通常 < embed_dim 或 > embed_dim)
            num_heads (int): 多头注意力的头数
            temper (float): 温度系数，用于缩放score
        """
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        print("use AGTStyleGraphAttention with hidden_size instead of GraphMultiHeadAttention")

        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.temper = temper

        # ---- (1) embed_dim -> hidden_size ----
        self.input_proj = nn.Linear(embed_dim, hidden_size)

        # ---- (2) AGT-style attention projections ----
        self.linear_l = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_r = nn.Linear(hidden_size, hidden_size, bias=False)

        self.att_l = nn.Linear(self.head_dim, 1, bias=False)
        self.att_r = nn.Linear(self.head_dim, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(config.get("leaky_relu", 0.01))

        # ---- (3) hidden_size -> embed_dim ----
        self.out_proj = nn.Linear(hidden_size, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x, bias=None):
        """
        Args:
            x (torch.Tensor): [B, L, N, C] 输入
            bias (torch.Tensor): [num_heads, N, N] (可选)
        Returns:
            torch.Tensor: [B, L, N, C] 输出
        """
        B, L, N, C = x.shape
        residual = x  # 残差

        # ---- 1. flatten B, L ----
        x_reshaped = x.view(B * L, N, C)

        # ---- 2. embed_dim -> hidden_size ----
        h = self.input_proj(x_reshaped)  # [B*L, N, hidden_size]

        # ---- 3. 线性投射 (AGT-style) ----
        fl = self.linear_l(h).reshape(B * L, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B*L, H, N, D_h]
        fr = self.linear_r(h).reshape(B * L, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B*L, H, N, D_h]

        # ---- 4. 注意力分数 ----
        score_l = self.att_l(self.leaky_relu(fl))                           # [B*L, H, N, 1]
        score_r = self.att_r(self.leaky_relu(fr)).permute(0, 1, 3, 2)       # [B*L, H, 1, N]
        attn_scores = score_l + score_r                                     # [B*L, H, N, N]

        if bias is not None:
            attn_scores = attn_scores + bias.unsqueeze(0)

        attn_scores = attn_scores / self.temper
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # ---- 5. 加权求和 ----
        context = torch.matmul(attn_weights, fr)  # [B*L, H, N, D_h]

        # ---- 6. 合并多头 ----
        context = context.transpose(1, 2).reshape(B * L, N, self.hidden_size)

        # ---- 7. hidden_size -> embed_dim ----
        output = self.out_proj(context)
        output = output.view(B, L, N, C)

        # ---- 8. 残差 + LayerNorm ----
        output = self.ln(residual + output)

        return output

def extend_feature(node_features, intersection_groups):
    """
    Extends the feature matrix for a batched, time-series input, where the set of
    N nodes is THE SAME for every batch.

    Args:
        node_features (torch.Tensor): Feature tensor with shape (B, T, N, C).
        intersection_groups (list of lists): T-node groups with DIRECTLY usable indices (0 to N-1).
    """
    # Get dimensions. B=Batch, T=Time, N=Nodes, C=Channels
    B, T, N, C = node_features.shape
    num_X = len(intersection_groups)

    # If there are no intersection groups, no extension is needed.
    if num_X == 0:
        return node_features

    # This list will hold the feature tensors for each new 'X' node.
    # Each tensor will have shape (B, T, C).
    x_features_list = []

    # --- Loop only through the intersection groups ---
    # We no longer need to loop through the batch dimension.
    for group in intersection_groups:

        if not group:
            # If a group is empty, append a zero-feature tensor for this X node.
            # It will have the correct shape for the whole batch and all time steps.
            x_features_list.append(torch.zeros((B, T, C),
                                               dtype=node_features.dtype,
                                               device=node_features.device))
            continue

        # 1. Gather features for the current group across the ENTIRE batch and ALL time steps.
        # Indexing on the N dimension (dim=2).
        # Shape of t_node_features: (B, T, len(group), C)
        t_node_features = node_features[:, :, group, :]

        # 2. Average across the group's node dimension (dim=2).
        # The result is the feature for ONE 'X' node, for the whole batch and all time steps.
        # Shape of x_node_feature: (B, T, C)
        x_node_feature = torch.mean(t_node_features, dim=2)

        x_features_list.append(x_node_feature)

    # 3. Stack the new X-node features along a new dimension.
    # The list has num_X tensors of shape (B, T, C).
    # Stacking on dim=2 creates a single tensor of all X-node features.
    # Shape of x_features: (B, T, num_X, C)
    x_features = torch.stack(x_features_list, dim=2)

    # 4. Concatenate the original features with the new X-node features along the Node dimension (dim=2).
    # (B, T, N, C) + (B, T, num_X, C) -> (B, T, N + num_X, C)
    extended_features = torch.cat([node_features, x_features], dim=2)

    return extended_features

class spatialTransformer(nn.Module):
    def __init__(self, model_dim, x_hdim, g_hdim, num_relations, num_RGCN_layers, relation_mx, extended_mx, num_R, num_T,
                 num_X, intersection_groups, trajDist_mx, flowCnt_mx,
                 num_heads, gma_version,
                 config) -> None:
        super(spatialTransformer, self).__init__()

        self.use_RGCN = config.get("use_RGCN", False)
        self.use_E1 = config.get("use_E1", False)
        self.use_E2 = config.get("use_E2", False)
        self.use_E3 = config.get("use_E3", False)

        self.relation_mx = relation_mx
        self.intersection_groups = intersection_groups
        self.num_R = num_R
        self.num_T = num_T
        self.num_X = num_X
        self.extended_mx = extended_mx

        self.trajDist_mx = trajDist_mx
        self.flowCnt_mx = flowCnt_mx
        if not self.use_E1:
            # 设为只有一种关系即连接关系
            num_relations = 1
            extended_mx[extended_mx > 0] = 1
        if not self.use_E2:
            self.trajDist_mx.fill(np.inf)
        if not self.use_E3:
            self.flowCnt_mx.fill(0.0)

        self.HRGCN = HierarchicalRGCN(model_dim, g_hdim, model_dim,
                                      num_relations, num_RGCN_layers, extended_mx, num_R, num_T, num_X,
                                      intersection_groups)
        self.graph_bias_gen = GraphBiasModule(hidden_dim=num_heads)
        # self.multiHeadTransformer = GraphMultiHeadAttention(model_dim, num_heads, config, gma_version)
        self.multiHeadTransformer = AGTStyleGraphAttention(model_dim, x_hdim, num_heads, config)

    def forward(self, x):
        if np.any(self.extended_mx) and self.use_RGCN:
            # x = self.RGCN(x)
            # print("x", x.shape)
            extended_x = extend_feature(x, self.intersection_groups)
            # print("extended_x", extended_x.shape)
            x = self.HRGCN(extended_x)
        B, L, N, C = x.shape
        E2 = torch.from_numpy(self.trajDist_mx).float().to(x.device)
        E3 = torch.from_numpy(self.flowCnt_mx).float().to(x.device)
        bias = self.graph_bias_gen(E2, E3)
        return self.multiHeadTransformer(x, bias)

# class GATv2AttentionBlock(nn.Module):
#     """一个GATv2注意力块, 包含 MHA + Add&Norm + FFN + Add&Norm"""
#
#     def __init__(self, embed_dim, num_heads, dropout=0.1):
#         super().__init__()
#         assert embed_dim % num_heads == 0
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#
#         self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
#
#         self.leaky_relu = nn.LeakyReLU(0.2)
#         self.attn_weights_proj = nn.Linear(self.head_dim * 2, self.head_dim)
#         self.attn_vec = nn.Parameter(torch.Tensor(1, self.num_heads, 1, self.head_dim))
#         nn.init.xavier_uniform_(self.attn_vec.data, gain=1.414)
#
#         self.attn_dropout = nn.Dropout(dropout)
#         self.resid_dropout = nn.Dropout(dropout)
#
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim * 4),
#             nn.GELU(),
#             nn.Linear(embed_dim * 4, embed_dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         # x shape: [B_new, L, C] where B_new = B*N
#         B_new, L, C = x.shape
#
#         # --- Multi-Head GATv2 Attention ---
#         residual = x
#         x = self.norm1(x)
#
#         qkv = self.qkv_proj(x).chunk(3, dim=-1)
#         # GATv2 中 K 和 Q 是相同的，都等于 V
#         v = qkv[2].reshape(B_new, L, self.num_heads, self.head_dim).transpose(1, 2)
#
#         v_i = v.unsqueeze(3).expand(-1, -1, -1, L, -1)  # [B_new, H, L, L, D_h]
#         v_j = v.unsqueeze(2).expand(-1, -1, L, -1, -1)  # [B_new, H, L, L, D_h]
#
#         all_pairs = torch.cat([v_i, v_j], dim=-1)
#         attn_logits = self.leaky_relu(self.attn_weights_proj(all_pairs))
#         attn_logits = (attn_logits * self.attn_vec.unsqueeze(3)).sum(dim=-1)  # [B_new, H, L, L]
#
#         attn_weights = F.softmax(attn_logits, dim=-1)
#         attn_weights = self.attn_dropout(attn_weights)
#
#         context = torch.matmul(attn_weights.unsqueeze(3), v.unsqueeze(2)).squeeze(3)
#         context = context.transpose(1, 2).reshape(B_new, L, self.embed_dim)
#         context = self.out_proj(context)
#
#         # --- Add & Norm + FFN + Add & Norm ---
#         x = residual + self.resid_dropout(context)
#
#         residual = x
#         x = self.norm2(x)
#         x = residual + self.ffn(x)
#
#         return x

class GATv2AttentionBlock(nn.Module):
    """GATv2 注意力块 (内部使用 hidden_dim, 输入输出维持 embed_dim)"""

    def __init__(self, embed_dim, num_heads, hidden_dim=None, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim  # 若不指定，则与 embed_dim 相同
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // num_heads

        # --- 投影层：输入embed_dim → hidden_dim ---
        self.input_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.qkv_proj = nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        self.out_proj = nn.Linear(self.hidden_dim, embed_dim)

        # --- 注意力参数 ---
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.attn_weights_proj = nn.Linear(self.head_dim * 2, self.head_dim)
        self.attn_vec = nn.Parameter(torch.Tensor(1, self.num_heads, 1, self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec.data, gain=1.414)

        # --- Dropout & Norm & FFN ---
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # FFN仍在 embed_dim 空间中工作（与Transformer保持一致）
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x: [B_new, L, embed_dim]
        return: [B_new, L, embed_dim]
        """
        B_new, L, _ = x.shape

        # --- Add & Norm 前 ---
        residual = x
        x = self.norm1(x)

        # --- 投影到 hidden_dim 空间 ---
        x_hidden = self.input_proj(x)  # [B, L, hidden_dim]

        # --- GATv2 注意力 ---
        qkv = self.qkv_proj(x_hidden).chunk(3, dim=-1)
        v = qkv[2].reshape(B_new, L, self.num_heads, self.head_dim).transpose(1, 2)

        v_i = v.unsqueeze(3).expand(-1, -1, -1, L, -1)
        v_j = v.unsqueeze(2).expand(-1, -1, L, -1, -1)
        all_pairs = torch.cat([v_i, v_j], dim=-1)

        attn_logits = self.leaky_relu(self.attn_weights_proj(all_pairs))
        attn_logits = (attn_logits * self.attn_vec.unsqueeze(3)).sum(dim=-1)

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights.unsqueeze(3), v.unsqueeze(2)).squeeze(3)
        context = context.transpose(1, 2).reshape(B_new, L, self.hidden_dim)

        # --- 映射回 embed_dim 空间 ---
        context = self.out_proj(context)

        # --- 残差连接 + FFN ---
        x = residual + self.resid_dropout(context)
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x

class TemporalAttentionLayer(nn.Module):
    """
    可切换的Temporal Attention层，支持 'standard' 和 'gatv2'。
    输入和输出形状均为 (B*N, L, C)。
    """

    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1, attn_version='gatv2', config=None):
        super().__init__()
        self.attn_version = attn_version
        # self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len=config.get("input_window", 5000))
        self.pos_encoder = get_pos_encoder(config.get('pe_type', '3'))(embed_dim, dropout, max_len=config.get("input_window", 5000))

        if attn_version == 'standard':
            print("TemporalAttention使用standard版本")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
            )
            self.attention_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif attn_version == 'gatv2':
            print("TemporalAttention使用gatv2版本")
            t_hdim = config.get("t_hdim")
            self.attention_blocks = nn.ModuleList(
                [GATv2AttentionBlock(embed_dim, num_heads, hidden_dim=t_hdim, dropout=dropout) for _ in range(num_layers)]
            )
        else:
            raise ValueError(f"Unknown attention version: {attn_version}")

    def forward(self, x):
        # x shape: [B*N, L, C]
        x = self.pos_encoder(x)

        if self.attn_version == 'standard':
            return self.attention_blocks(x)
        elif self.attn_version == 'gatv2':
            for block in self.attention_blocks:
                x = block(x)
            return x

class temporalTransformer(nn.Module):
    def __init__(self, model_dim, time_intervals, config, num_heads=8, num_layers=3, dropout=0.1,
                 attn_version='gatv2') -> None:
        super().__init__()

        self.use_tf = config.get("use_tf", False)

        self.model_dim = model_dim
        self.time_of_day_size = int((24 * 60 * 60) / time_intervals)
        self.day_of_week_size = 7
        self.tod_embedding_dim = config.get("tod_embedding_dim", 24)
        self.dow_embedding_dim = config.get("dow_embedding_dim", 24)
        self.avgSpeed_embedding_dim = config.get("avgSpeed_embedding_dim", 24)

        # 1. 定义嵌入层
        self.tod_embedding = nn.Embedding(self.time_of_day_size, self.tod_embedding_dim)
        self.dow_embedding = nn.Embedding(self.day_of_week_size, self.dow_embedding_dim)
        self.avgSpeed_embedding = nn.Linear(1, self.avgSpeed_embedding_dim)

        self.tf_first = config.get("tf_first", 0)

        # 2. 定义可切换的注意力模块
        if self.use_tf and self.tf_first == 0:
            hidden_dim = model_dim + self.tod_embedding_dim + self.dow_embedding_dim + self.avgSpeed_embedding_dim
        else:
            hidden_dim = model_dim
        self.attention_layer = TemporalAttentionLayer(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            attn_version=attn_version,  # 关键参数
            config=config
        )

        # 3. 定义最终的回归层
        self.regression_layer = nn.Linear(hidden_dim, model_dim)
        self.use_ff = False
        if config.get("use_ff", False):
            self.use_ff = True
            print(int(self.use_ff), "use_feed_forward")
            feed_forward_dim = config.get("feed_forward_dim", 256)
            self.feed_forward = nn.Sequential(
                nn.Linear(hidden_dim, feed_forward_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feed_forward_dim, model_dim),
            )

    def forward(self, input_data: torch.Tensor, tid_data, diw_data, avg_speeds) -> torch.Tensor:
        # --- (A) 特征工程部分 (保持不变) ---
        batch_size, seq_len, num_nodes, _ = input_data.shape
        device = input_data.device

        # 时间嵌入
        tid_indices = (tid_data * self.time_of_day_size).long()
        time_in_day_emb = self.tod_embedding(tid_indices)

        # 星期嵌入
        diw_indices = diw_data.long()
        day_in_week_emb = self.dow_embedding(diw_indices)

        # 历史速度嵌入
        indices = (diw_data * 96 + tid_data).long()
        historical_speeds = torch.from_numpy(avg_speeds).float().to(device)
        past_speeds_values = historical_speeds[indices, torch.arange(num_nodes, device=device).view(1, 1, -1)]
        past_speeds = past_speeds_values.unsqueeze(-1)
        avg_speed_emb = self.avgSpeed_embedding(past_speeds)

        # 拼接所有特征
        if self.use_tf and self.tf_first == 0:
            x = torch.cat([input_data, time_in_day_emb, day_in_week_emb, avg_speed_emb], dim=-1)
        else:
            x = input_data

        # --- (B) 维度变换与注意力计算 ---
        B, L, N, C = x.shape
        # [B, L, N, C] -> [B*N, L, C]
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, L, C)

        # 调用可切换的注意力层
        output_reshaped = self.attention_layer(x_reshaped)

        # [B*N, L, C] -> [B, L, N, C]
        output = output_reshaped.view(B, N, L, C).permute(0, 2, 1, 3)

        # --- (C) 回归输出 ---
        if self.use_ff:
            output = self.feed_forward(output)
        else:
            output = self.regression_layer(output)

        return output

class ST_Block(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, model_dim, x_hdim, g_hdim, num_relations, num_RGCN_layers,
                 relation_mx, extended_mx, num_R, num_T, num_X, intersection_groups,
                 trajDist_mx, flowCnt_mx,
                 num_spa_heads, num_temp_heads, gma_version, time_intervals,
                 config) -> None:
        super().__init__()
        self.use_spaXformer = config.get("use_spaXformer", False)
        self.use_tempXformer = config.get("use_tempXformer", False)
        self.HR_reversed = config.get("HR_reversed", 0)
        self.use_resST = config.get("use_resST", False)
        if self.use_resST:
            self.ln = nn.LayerNorm(model_dim)
        tf_layers = config.get("tf_layers", 3)
        self.spatialTransformer = spatialTransformer(model_dim, x_hdim, g_hdim, num_relations, num_RGCN_layers,
                                                     relation_mx, extended_mx, num_R, num_T, num_X, intersection_groups,
                                                     trajDist_mx, flowCnt_mx,
                                                     num_spa_heads, gma_version,
                                                     config)
        self.temporalTransformer = temporalTransformer(model_dim, time_intervals, config, num_heads=num_temp_heads,
                                                       num_layers=tf_layers)

    def forward(self, input_data, tid_data, diw_data, avgSpeed_tod_dow) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, L, N, C]
            tid_data (torch.Tensor): input data with shape [B, L, N, C]
            diw_data (torch.Tensor): input data with shape [B, L, N, C]

        Returns:
            torch.Tensor: latent repr
        """
        hidden = input_data
        if bool(self.HR_reversed):
            if self.use_tempXformer:
                hidden = self.temporalTransformer(hidden, tid_data, diw_data, avgSpeed_tod_dow)
                if self.use_resST:
                    residual = hidden
            if self.use_spaXformer:
                hidden = self.spatialTransformer(hidden)
                if self.use_resST:
                    hidden = self.ln(residual + hidden)
        else:
            if self.use_spaXformer:
                hidden = self.spatialTransformer(hidden)
                if self.use_resST:
                    residual = hidden
            if self.use_tempXformer:
                hidden = self.temporalTransformer(hidden, tid_data, diw_data, avgSpeed_tod_dow)
                if self.use_resST:
                    hidden = self.ln(residual + hidden)

        return hidden

class HIT_Former(AbstractTrafficStateModel):
    """
    Paper: MR-STHGformer: Multi-Relation Spatial Temporal Heterogeneous Graph Transformer for Urban Traffic Prediction
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.num_nodes = data_feature.get('num_nodes')
        self.input_window = config.get('input_window')
        self.output_window = config.get('output_window')
        self.feature_dim = data_feature.get('feature_dim', 2)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.time_intervals = config.get('time_intervals')
        self._scaler = self.data_feature.get('scaler')

        self.num_block = config.get('num_block')

        self.device = config.get('device', torch.device('cpu'))

        self.num_relations = data_feature.get('num_relations')
        self.num_RGCN_layers = config.get('num_RGCN_layers')
        self.relation_mx = data_feature.get('relation_mx')
        self.extended_mx = data_feature.get('extended_mx')
        self.num_R = data_feature.get('num_R')
        self.num_T = data_feature.get('num_T')
        self.num_X = data_feature.get('num_X')
        self.intersection_groups = data_feature.get('intersection_groups')

        self.trajDist_mx = data_feature.get('trajDist_mx')
        self.flowCnt_mx = data_feature.get('flowCnt_mx')
        self.num_spa_heads = config.get('num_spa_heads', 8)
        self.num_temp_heads = config.get('num_temp_heads', 8)
        self.gma_version = config.get('gma_version', 'gatv2')
        self.road_feature_dim = config.get('road_feature_dim', 1)
        self.avgSpeed_tod_dow = data_feature.get('avgSpeed_tod_dow')

        self.input_embedding_dim = config.get("input_embedding_dim", 24)  # D
        self.tod_embedding_dim = config.get("tod_embedding_dim", 24)
        self.dow_embedding_dim = config.get("dow_embedding_dim", 24)
        self.avgSpeed_embedding_dim = config.get("avgSpeed_embedding_dim", 24)
        self.road_feature_embedding_dim = config.get("road_feature_embedding_dim", 80)
        self.x_hdim = config.get("x_hdim", 32)
        self.g_hdim = config.get("g_hdim", 32)



        assert (24 * 60 * 60) % self.time_intervals == 0, "time_of_day_size should be Int"
        self.time_of_day_size = int((24 * 60 * 60) / self.time_intervals)
        self.day_of_week_size = 7

        self._logger = getLogger()

        self.input_proj = nn.Linear(1, self.input_embedding_dim)

        self.road_feature_dim = data_feature.get("road_feature_dim", 0)
        self.road_feature_mx = data_feature.get("road_feature_mx")
        self.tf_first = config.get("tf_first", 0)
        self.model_dim = self.input_embedding_dim
        if self.road_feature_dim != 0 and np.any(self.road_feature_mx) and self.tf_first == 0:
            print("road_feature_dim", self.road_feature_dim)
            print("road_feature_mx", self.road_feature_mx.shape, self.road_feature_mx[:6, :self.road_feature_dim])
            print("speed and road features are transferred to 2 different linear layers")

            self.HFE = nn.Linear(self.road_feature_dim, self.road_feature_embedding_dim)

            self.model_dim = (self.input_embedding_dim
                              + self.road_feature_embedding_dim
                              )

        elif self.road_feature_dim != 0 and np.any(self.road_feature_mx) and self.tf_first == 1:
            print("road_feature_dim", self.road_feature_dim)
            print("road_feature_mx", self.road_feature_mx.shape, self.road_feature_mx[:6, :self.road_feature_dim])
            print("speed and road features are transferred to 2 different linear layers")
            print("use tf as the input before the st-blocks")

            self.HFE = nn.Linear(self.road_feature_dim, self.road_feature_embedding_dim)

            self.tod_embedding = nn.Embedding(self.time_of_day_size, self.tod_embedding_dim)
            self.dow_embedding = nn.Embedding(self.day_of_week_size, self.dow_embedding_dim)
            self.avgSpeed_embedding = nn.Linear(1, self.avgSpeed_embedding_dim)

            self.model_dim = (self.input_embedding_dim
                              + self.tod_embedding_dim
                              + self.dow_embedding_dim
                              + self.avgSpeed_embedding_dim
                              + self.road_feature_embedding_dim
                              )

        print("model_dim", self.model_dim)

        self.STblocks = nn.ModuleList([
            ST_Block(self.model_dim, self.x_hdim, self.g_hdim, self.num_relations, self.num_RGCN_layers,
                     self.relation_mx, self.extended_mx, self.num_R, self.num_T, self.num_X, self.intersection_groups,
                     self.trajDist_mx, self.flowCnt_mx,
                     self.num_spa_heads, self.num_temp_heads, self.gma_version, self.time_intervals,
                     config) for _ in
            range(self.num_block)
        ])
        # regression
        self.regression_layer = nn.Linear(self.model_dim, self.output_dim)

    def forward(self, batch):
        # prepare data
        input_data = batch['X']  # [B, L, N, C] batch, length, num_nodes, features
        B, L, N, C = input_data.shape
        rf_time_series = input_data[..., :1]  # [B, L, N, 1]

        tod = input_data[..., 1]  # [B, L, N]
        dow = torch.argmax(input_data[..., 2:], dim=-1)  # [B, L, N]

        # time series embedding
        x = self.input_proj(rf_time_series)  # [B, L, N, 1] -> [B, L, N, 24]
        features = [x]
        if self.road_feature_dim != 0 and np.any(self.road_feature_mx) and self.tf_first == 0:
            rf_series = torch.from_numpy(self.road_feature_mx).unsqueeze(0).unsqueeze(1).expand(B, L, -1,
                                                                                                -1)  # [B, L, N, rf]
            rf_embedding = self.HFE(rf_series.to(self.device))  # [B, L, N, rf] -> [B, L, N, 80]
            features.append(rf_embedding)
        elif self.road_feature_dim != 0 and np.any(self.road_feature_mx) and self.tf_first == 1:
            rf_series = torch.from_numpy(self.road_feature_mx).unsqueeze(0).unsqueeze(1).expand(B, L, -1,
                                                                                                -1)  # [B, L, N, rf]
            rf_embedding = self.HFE(rf_series.to(self.device))  # [B, L, N, rf] -> [B, L, N, 80]
            features.append(rf_embedding)

            # 时间嵌入
            tod_emb = self.tod_embedding(
                (tod * self.time_of_day_size).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)

            # 星期嵌入
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)

            # 历史速度嵌入
            indices = (tod * 96 + dow).long()
            avg_speeds = self.avgSpeed_tod_dow
            historical_speeds = torch.from_numpy(avg_speeds).float().to(self.device)
            past_speeds_values = historical_speeds[
                indices, torch.arange(self.num_nodes, device=self.device).view(1, 1, -1)]
            past_speeds = past_speeds_values.unsqueeze(-1)
            avg_speed_emb = self.avgSpeed_embedding(past_speeds)
            features.append(avg_speed_emb)

        hidden = torch.cat(features, dim=-1)

        for block in self.STblocks:
            hidden = block(hidden.to(self.device), tod.to(self.device), dow.to(self.device),
                           self.avgSpeed_tod_dow)  # hidden 逐层更新

        prediction = self.regression_layer(hidden)

        return prediction

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
