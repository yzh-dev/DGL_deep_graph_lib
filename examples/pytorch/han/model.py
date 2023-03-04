import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )
    def forward(self, z):
        w = self.project(z).mean(0)  # (2, 1)，对所有节点求均值
        beta = torch.softmax(w, dim=0)  # (2, 1) 2个meta-path进行归一化
        beta = beta.expand((z.shape[0],) + beta.shape)  # 拓展到N个节点上(3025, 2, 1)
        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(
        self, num_meta_paths, in_size, out_size, layer_num_heads, dropout
    ):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):#2个meta-path
            self.gat_layers.append(#对每个meta-path形成的异构图，采用GAT计算节点特征
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                )
            )
        self.semantic_attention = SemanticAttention(#语义attention，其实就是全连接
            in_size=out_size * layer_num_heads
        )
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []
        # 遍历每个meta-path下的图
        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))#gat_layers[i](g, h) g:图; h:features => [3025, 8, 8] => [3025, 64]
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # [3025, 2, 64]每个节点对应到metapath下的每个节点的embedding值（Node Attention）
        # 聚合meta-path下，每个节点最终的输出值
        return self.semantic_attention(semantic_embeddings)  #  [3025, 64]


class HAN(nn.Module):
    def __init__(
        self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(
                num_meta_paths, in_size, hidden_size, num_heads[0], dropout
            )
        )
        for l in range(1, len(num_heads)):#多层多头，目前只有1层异构图
            self.layers.append(
                HANLayer(
                    num_meta_paths,
                    hidden_size * num_heads[l - 1],#前一层的维度：隐藏层维度hidden_size*heads
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)#HANLayer

        return self.predict(h)
