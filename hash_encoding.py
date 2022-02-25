import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from utils import get_voxel_vertices, get_plane_vertices

class HashEmbedder(nn.Module):
    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level

        self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))

        self.embeddings = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size, \
                                        self.n_features_per_level) for i in range(n_levels)])
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()
        

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,4]*weights[:,0][:,None]
        c01 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,5]*weights[:,0][:,None]
        c10 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,6]*weights[:,0][:,None]
        c11 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,7]*weights[:,0][:,None]

        # step 2
        c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]
        c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]

        # step 3
        c = c0*(1-weights[:,2][:,None]) + c1*weights[:,2][:,None]

        return c

    def forward(self, x):
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = get_voxel_vertices(\
                                                x, self.bounding_box, \
                                                resolution, self.log2_hashmap_size)
            
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)

            x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        return torch.cat(x_embedded_all, dim=-1)

class HashTriEmbedder(nn.Module):
    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashTriEmbedder, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level * 3

        self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))

        #self.embeddings = [None]*3
        #for i in range(3):
        #    self.embeddings[i] = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size, \
        #                                    self.n_features_per_level) for i in range(n_levels)] )
        self.embeddings_xy = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size, \
                                            self.n_features_per_level) for i in range(n_levels)] )
        self.embeddings_xz = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size, \
                                            self.n_features_per_level) for i in range(n_levels)] )
        self.embeddings_yz = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size, \
                                            self.n_features_per_level) for i in range(n_levels)] )
        self.embeddings = [self.embeddings_xy, self.embeddings_xz, self.embeddings_yz]

        # custom uniform initialization
        for i0 in range(3):
            for i1 in range(n_levels):
                nn.init.uniform_(self.embeddings[i0][i1].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()

    def bilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 2
        voxel_min_vertex: B x 2
        voxel_max_vertex: B x 2
        voxel_embedds: B x 4 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 2

        # step 1
        # 0->00, 1->01, 2->10, 4->11
        c0 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,2]*weights[:,0][:,None]
        c1 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,3]*weights[:,0][:,None]

        # step 3
        c = c0*(1-weights[:,1][:,None]) + c1*weights[:,1][:,None]

        return c
    
    def separate_input(self,x):
        in_xy = x[:,:2]
        in_xz = x[:,::2]
        in_yz = x[:,-2:]
        return [in_xy,in_xz,in_yz]


    def forward(self, x):
        # x is 3D point position: B x 3
        x_sep = self.separate_input(x)
        x_embedded_all = []
        for i in range(self.n_levels):  #16
            resolution = torch.floor(self.base_resolution * self.b**i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = get_plane_vertices(\
                                                x, self.bounding_box, \
                                                resolution, self.log2_hashmap_size) #[3*(B, 2)]  [3*(B, 2)]  [3*(B, 4)]
            
            for j in range(3):
                voxel_embedds = self.embeddings[j][i](hashed_voxel_indices[j])     # (B, 4, 2)
                x_embedded = self.bilinear_interp(x_sep[j], voxel_min_vertex[j], voxel_max_vertex[j], voxel_embedds)
                # (B, 2)
                x_embedded_all.append(x_embedded)

            #pdb.set_trace()
        return torch.cat(x_embedded_all, dim=-1)

class TriEmbedder(nn.Module):
    def __init__(self, bounding_box, n_levels=16, n_features_per_level=16,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(TriEmbedder, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level * 3 

        self.embeddings_xy = nn.Embedding(self.finest_resolution**2, self.n_features_per_level)
        self.embeddings_xz = nn.Embedding(self.finest_resolution**2, self.n_features_per_level) 
        self.embeddings_yz = nn.Embedding(self.finest_resolution**2, self.n_features_per_level)
        self.embeddings = nn.ModuleList([self.embeddings_xy, self.embeddings_xz, self.embeddings_yz])

        # custom uniform initialization
        for i0 in range(3):
            nn.init.uniform_(self.embeddings[i0].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()

    def bilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 2
        voxel_min_vertex: B x 2
        voxel_max_vertex: B x 2
        voxel_embedds: B x 4 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 2

        # step 1
        # 0->00, 1->01, 2->10, 4->11
        c0 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,2]*weights[:,0][:,None]
        c1 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,3]*weights[:,0][:,None]

        # step 3
        c = c0*(1-weights[:,1][:,None]) + c1*weights[:,1][:,None]

        return c
    
    def separate_input(self,x):
        in_xy = x[:,:2]
        in_xz = x[:,::2]
        in_yz = x[:,-2:]
        return [in_xy,in_xz,in_yz]

    def forward(self, x):
        # x is 3D point position: B x 3
        x_sep = self.separate_input(x)
        x_embedded_all = []
        
        #resolution = torch.floor(self.base_resolution * self.b**i)
        resolution = self.finest_resolution
        voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = get_plane_vertices(\
                                            x, self.bounding_box, \
                                            resolution, self.log2_hashmap_size) #[3*(B, 2)]  [3*(B, 2)]  [3*(B, 4)]
        
        for j in range(3):
            voxel_embedds = self.embeddings[j](hashed_voxel_indices[j])     # (B, 4, 2)
            x_embedded = self.bilinear_interp(x_sep[j], voxel_min_vertex[j], voxel_max_vertex[j], voxel_embedds)
            # (B, 2)
            x_embedded_all.append(x_embedded)

            #pdb.set_trace()
        return torch.cat(x_embedded_all, dim=-1)

class SHEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4):
    
        super().__init__()

        self.input_dim = input_dim
        self.degree = degree

        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        self.out_dim = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input, **kwargs):

        result = torch.empty((*input.shape[:-1], self.out_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                #result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result
