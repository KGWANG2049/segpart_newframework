import torch
from torch import nn
from utils import ops
from models.attention_variants import Global_based_attention_variation
from models.pe_attention_variants import Pe_global_based_attention_variation


class SelfAttention(nn.Module):
    def __init__(self, pe_method, q_in=64, q_out=64, k_in=64, k_out=64, v_in=64, v_out=64, num_heads=8,
                 att_score_method='global_dot'):
        super(SelfAttention, self).__init__()
        # check input values
        if k_in != v_in:
            raise ValueError(f'k_in and v_in should be the same! Got k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out != v_out:
            raise ValueError('Please check the dimension of energy')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')
        self.att_score_method = att_score_method
        self.num_heads = num_heads
        self.pe_method = pe_method
        self.q_depth = int(q_out / num_heads)
        self.k_depth = int(k_out / num_heads)
        self.v_depth = int(v_out / num_heads)
        if pe_method == 'false' or pe_method == 'pe_i':
            self.Global_based_attention_variation = Global_based_attention_variation()
        elif pe_method == 'pe_ii' or pe_method == 'pe_iii' or pe_method == 'pe_iv':
            self.Pe_global_based_attention_variation = Pe_global_based_attention_variation(pe_method, num_heads, k_out)

        self.q_conv = nn.Conv1d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv1d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv1d(v_in, v_out, 1, bias=False)

    def forward(self, x, xyz=None):  # x.shape == (B, C, N)

        q = self.q_conv(x)  # q.shape == (B, C, N)
        q = self.global_split_heads(q, self.num_heads, self.q_depth)  # q.shape == (B, H, N, D)
        k = self.k_conv(x)  # k.shape ==  (B, C, N)
        k = self.global_split_heads(k, self.num_heads, self.k_depth)  # k.shape == (B, H, N, D)
        v = self.v_conv(x)  # v.shape ==  (B, C, N)
        v = self.global_split_heads(v, self.num_heads, self.v_depth)  # v.shape == (B, H, N, D)

        if self.att_score_method == 'global_dot':
            if self.pe_method == 'false' or self.pe_method == 'pe_i':
                x = self.Global_based_attention_variation.global_attention_Dot(q, k, v)
            elif self.pe_method == 'pe_ii' or self.pe_method == 'pe_iii' or self.pe_method == 'pe_iv':
                x = self.Pe_global_based_attention_variation.global_attention_Dot(q, k, v, xyz)

        elif self.att_score_method == 'global_sub':
            if self.pe_method == 'false' or self.pe_method == 'pe_i':
                x = self.Global_based_attention_variation.global_attention_Sub(q, k, v)
            elif self.pe_method == 'pe_ii' or self.pe_method == 'pe_iii' or self.pe_method == 'pe_iv':
                x = self.Pe_global_based_attention_variation.global_attention_Sub(q, k, v, xyz)

        else:
            raise ValueError(f"att_score_method must be 'global_dot', 'global_sub'. Got: {self.att_score_method}")

        x = x.reshape(x.shape[0], -1, x.shape[1])  # x.shape == (B, C, N)
        return x

    def global_split_heads(self, x, heads, depth):  # x.shape == (B, C, N)
        x = x.view(x.shape[0], heads, depth, x.shape[2])  # x.shape == (B, H, D, N)
        x = x.permute(0, 1, 3, 2)  # x.shape == (B, H, N, D)
        return x


class Global_SelfAttention_Layer(nn.Module):
    def __init__(self, pe_method, q_in, q_out, k_in, k_out, v_in, v_out, num_heads, att_score_method,
                 ff_conv1_channels_in, ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out):
        super(Global_SelfAttention_Layer, self).__init__()

        if q_in != v_out:
            raise ValueError(f'q_in should be equal to v_out due to ResLink! Got q_in: {q_in}, v_out: {v_out}')
        self.att_score_method = att_score_method
        self.pe_method = pe_method
        self.sa = SelfAttention(pe_method, q_in, q_out, k_in, k_out, v_in, v_out, num_heads, att_score_method)
        self.ff = nn.Sequential(nn.Conv1d(ff_conv1_channels_in, ff_conv1_channels_out, 1, bias=False),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(ff_conv2_channels_in, ff_conv2_channels_out, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(v_out)
        self.bn2 = nn.BatchNorm1d(v_out)

    def forward(self, pcd, xyz=None):
        x_out = self.sa(pcd, xyz)  # x_out.shape == (B, C, N)

        x = self.bn1(pcd + x_out)  # x.shape == (B, C, N)
        x_out = self.ff(x)  # x_out.shape == (B, C, N)
        x = self.bn2(x + x_out)  # x.shape == (B, C, N)
        return x


"""class Downsample(nn.Module):
    def __init__(self, n_samples, embedding_k, conv1_channel_in, conv1_channel_out, conv2_channel_in,
                 conv2_channel_out):
        super(Downsample, self).__init__()
        self.n_samples = n_samples
        self.edgeConv = ops.EdgeConv(embedding_k, conv1_channel_in, conv1_channel_out, conv2_channel_in,
                                     conv2_channel_out)

    def forward(self, x):  # x.shape == (B, C, N)
        x_downsampled, idx = ops.fps(x, self.n_samples)  # Get both sampled points and their indices
        x_processed = self.edgeConv(x_downsampled, x)

        return x_processed, idx  # Return both processed points and their idx"""


class Downsample(nn.Module):
    def __init__(self, points_select_basis, n_samples, embedding_k, conv1_channel_in, conv1_channel_out,
                 conv2_channel_in,
                 conv2_channel_out):
        super(Downsample, self).__init__()
        self.n_samples = n_samples
        self.points_select_basis = points_select_basis
        # self.points_select_basis = points_select_basis
        self.edgeConv = ops.EdgeConv(embedding_k, conv1_channel_in, conv1_channel_out, conv2_channel_in,
                                     conv2_channel_out)

    def forward(self, x=None, xyz=None, index=None):  # x.shape == (B, C, N)
        if self.points_select_basis == 'feature':
            x_downsampled, idx = ops.fps(x, self.n_samples)  # Get both sampled points and their indices
            x_processed = self.edgeConv(x_downsampled, x)

        elif self.points_select_basis == 'xyz':
            selected_points = torch.gather(xyz, 2, index.unsqueeze(1).expand(-1, 3, -1))

            x_downsampled, idx = ops.fps(selected_points, self.n_samples)  # Get both sampled points and their indices
            x_processed = self.edgeConv(x_downsampled, xyz)
        else:
            raise ValueError(f'points_select_basis should be equal to feature or xyz! But got: '
                             f'{self.points_select_basis}, v_out: {self.points_select_basis}')

        return x_processed, idx  # Return both processed points and their idx


class Point_Embedding(nn.Module):
    def __init__(self, point_emb1_in, point_emb1_out, point_emb2_in, point_emb2_out):
        super(Point_Embedding, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(point_emb1_in, point_emb1_out, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(point_emb1_out),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(point_emb2_in, point_emb2_out, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(point_emb2_out),
                                   nn.LeakyReLU(negative_slope=0.2))

    ''''
    init_a.shape == (B, 3, N)
    init_b.shape == (B, 3, N)
    '''

    def forward(self, a):
        B, N = a.shape[0], a.shape[2]
        pcd_idx = torch.arange(N, device=a.device).repeat(B, 1)
        x = self.conv1(a)  # x.shape == (B, C, N)
        x = self.conv2(x)  # x.shape == (B, C, N)
        return x, pcd_idx


class UpSampleInterpolation(nn.Module):
    def __init__(self, v_dense, up_k):
        super(UpSampleInterpolation, self).__init__()
        self.up_k = up_k
        self.conv = nn.Sequential(nn.Conv1d(3 * v_dense, v_dense, 1, bias=False), nn.BatchNorm1d(v_dense),
                                  nn.LeakyReLU(negative_slope=0.2))
        # self.skip_link = nn.Sequential(nn.Conv1d(skip_link_in, skip_link_out, 1, bias=False), nn.BatchNorm1d(
        # skip_link_out), nn.LeakyReLU(negative_slope=0.2))

    def forward(self, dense_points, sparse_points, pcd):
        device = pcd.device
        dense_points_data, dense_points_idx = dense_points
        sparse_points_data, sparse_points_idx = sparse_points

        dense_points_data = dense_points_data
        dense_points_idx = dense_points_idx
        sparse_points_data = sparse_points_data
        sparse_points_idx = sparse_points_idx

        dense_idx_exp = dense_points_idx.unsqueeze(1).expand(-1, 3, -1)
        dense_points_xyz = torch.gather(pcd, 2, dense_idx_exp)
        sparse_idx_exp = sparse_points_idx.unsqueeze(1).expand(-1, 3, -1)
        sparse_points_xyz = torch.gather(pcd, 2, sparse_idx_exp)

        # idx_select.shape == (B, C)  sparse_points.shape == (B, C, M)
        interpolated_points = self.interpolate(dense_points_xyz, sparse_points_xyz, sparse_points_data, K=self.up_k)
        x = torch.cat([dense_points_data, interpolated_points], dim=1)
        x = self.conv(x)
        return x, dense_points_idx

    def interpolate(self, dense_points_xyz, sparse_points_xyz, sparse_points, K=3):
        # sparse_points = self.conv(sparse_points)  # points_select_conv.shape == (B, C, M)
        neighbors, _, d_neighbors = ops.select_neighbors_interpolate(dense_points_xyz, sparse_points_xyz,
                                                                     sparse_points,
                                                                     K=K)
        # neighbors.shape == (B, C, N, K), idx.shape == (B, N, K)
        weights = 1.0 / (d_neighbors + 1e-8)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)  # weights.shape == (B, N, K)
        interpolated_points = torch.sum(neighbors * weights.unsqueeze(dim=1), dim=-1)

        # interpolated_points.shape == (B, C, N), dim=-1)
        return interpolated_points


class ShapeNetModelSeg(nn.Module):
    def __init__(self, point_emb1_in, point_emb1_out, point_emb2_in, point_emb2_out, points_select_basis, n_samples,
                 embedding_k,
                 conv1_channel_in, conv1_channel_out, conv2_channel_in, conv2_channel_out, pe_method, q_in, q_out, k_in,
                 k_out, v_in, v_out, num_heads,
                 att_score_method,
                 ff_conv1_channels_in, ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out,
                 v_dense, up_k):
        super(ShapeNetModelSeg, self).__init__()

        # Point Embedding layers
        self.Point_Embedding_list = nn.ModuleList([Point_Embedding(point_emb1_in, point_emb1_out, point_emb2_in,
                                                                   point_emb2_out) for
                                                   point_emb1_in, point_emb1_out, point_emb2_in, point_emb2_out in
                                                   zip(point_emb1_in,
                                                       point_emb1_out,
                                                       point_emb2_in,
                                                       point_emb2_out)])

        # Downsample layers
        # self.points_select_basis = points_select_basis
        self.downsample_list = nn.ModuleList(
            [Downsample(points_select_basis, n_samples, embedding_k, conv1_channel_in, conv1_channel_out,
                        conv2_channel_in, conv2_channel_out) for
             points_select_basis, n_samples, embedding_k,
             conv1_channel_in, conv1_channel_out, conv2_channel_in, conv2_channel_out
             in zip(points_select_basis, n_samples, embedding_k, conv1_channel_in, conv1_channel_out,
                    conv2_channel_in, conv2_channel_out)])
        self.points_select_basis = points_select_basis

        # Global Self-Attention Layers
        self.Global_SelfAttention_Layer_list = nn.ModuleList(
            [Global_SelfAttention_Layer(pe_method, q_in, q_out, k_in, k_out, v_in, v_out, num_heads, att_score_method,
                                        ff_conv1_channels_in, ff_conv1_channels_out, ff_conv2_channels_in,
                                        ff_conv2_channels_out) for pe_method, q_in, q_out, k_in, k_out, v_in, v_out,
             num_heads, att_score_method, ff_conv1_channels_in, ff_conv1_channels_out, ff_conv2_channels_in,
             ff_conv2_channels_out in zip(pe_method, q_in, q_out, k_in, k_out, v_in, v_out, num_heads, att_score_method,
                                          ff_conv1_channels_in, ff_conv1_channels_out, ff_conv2_channels_in,
                                          ff_conv2_channels_out)])

        # Upsampling layers
        self.upsample_list = nn.ModuleList([UpSampleInterpolation(v_dense, up_k) for
                                            v_dense, up_k in
                                            zip(v_dense, up_k)])

      
        self.conv = nn.Sequential(nn.Conv1d(32, 1024, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(1024),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.conv1 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Sequential(nn.Conv1d(3136, 1024, kernel_size=1, bias=False), nn.BatchNorm1d(1024),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.linear2 = nn.Sequential(nn.Conv1d(1024, 256, kernel_size=1, bias=False), nn.BatchNorm1d(256),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)
        self.conv4 = nn.Conv1d(256, 50, kernel_size=1, bias=False)

    def forward(self, pcd, category_id):
        x_list = []
        idx_select_list = []
        xyz = pcd
        ori_xyz = pcd
        idx = None

        # Point Embedding
        for embedding in self.Point_Embedding_list:
            pcd, idx = embedding(pcd)
        x_list.append(pcd)
        idx_select_list.append(idx)

        attention_layers_per_downsample = [1, 1, 1, 1]
        accumulated_attention_layers = 0

        for i, downsample in enumerate(self.downsample_list):  # x, xyz, index
            if self.points_select_basis[i] == 'feature':
                pcd, idx = downsample(pcd, None, idx)
                for _ in range(attention_layers_per_downsample[i]):
                    pcd = self.Global_SelfAttention_Layer_list[accumulated_attention_layers](pcd)
                    accumulated_attention_layers += 1
                x_list.append(pcd)
                idx_select_list.append(idx)
            elif self.points_select_basis[i] == 'xyz':
                xyz, idx = downsample(None, xyz, idx)
                pcd = xyz
                for _ in range(attention_layers_per_downsample[i]):
                    pcd = self.Global_SelfAttention_Layer_list[accumulated_attention_layers](pcd)
                    accumulated_attention_layers += 1

                x_list.append(pcd)
                idx_select_list.append(idx)

        # Upsample
        x = (x_list.pop(), idx_select_list.pop())
        j = 0
        for upsample in self.upsample_list:
            pcd_down = (x_list.pop(), idx_select_list.pop())
            x, den_idx = upsample(pcd_down, x, ori_xyz)  # input == dense_points, sparse_points, ori-xyz

            if j < len(self.upsample_list) - 1:
                x = (x, den_idx)
            j += 1
        B, C, N = x.shape
        # x.shape == (B, 32, N)
        x_exp = self.conv(x)
        # x.shape == (B, 1024, N)
        category_id = self.conv1(category_id)
        # category_id.shape == (B, 64, N)
        x_max = x_exp.max(dim=-1, keepdim=True)[0]
        # x_max.shape == (B, 1024, 1)
        x_average = x_exp.mean(dim=-1, keepdim=True)
        # x_average.shape == (B, 1024, 1)
        x_global = torch.cat([x_max, x_average, category_id], dim=1)
        # x.shape === (B, 1024+1024+64, 1)
        x_global = x_global.repeat(1, 1, N)
        # x.shape == (B, 1024+1024+64, N)
        x = torch.cat([x_exp, x_global], dim=1)
        # x.shape == (B, 1024+2112, N)
        x = self.linear1(x)
        # x.shape == (B, 512, N)
        x = self.dp1(x)
        # x.shape == (B, 512, N)
        x = self.linear2(x)
        # x.shape == (B, 256, N)
        x = self.dp2(x)
        # x.shape == (B, 256, N)
        x = self.conv4(x)
        # x.shape == (B, 50, N)
        return x
