import json
import numpy as np
import pdb
import torch

from ray_utils import get_rays, get_ray_directions, get_ndc_rays


BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                               device='cuda')
SQR_OFFSETS = torch.tensor([[[i,j] for i in [0, 1] for j in [0, 1] ]], device='cuda')

def hash(coords, log2_hashmap_size):
    '''
    coords: 3D coordinates. B x 3
    log2T:  logarithm of T w.r.t 2
    '''
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    return torch.tensor((1<<log2_hashmap_size)-1) & (x*73856093 ^ y*19349663 ^ z*83492791)
    #return ((1<<log2_hashmap_size)-1) & (x*73856093 ^ y*19349663 ^ z*83492791)

def hash2d(coords, log2_hashmap_size):
    '''
    coords: 2D coordinates. B x 3
    log2T:  logarithm of T w.r.t 2
    '''
    x, y = coords[..., 0], coords[..., 1]
    return torch.tensor((1<<log2_hashmap_size)-1) & (x*73856093 ^ y*19349663)

def xy2index(xy,resolution):
    return xy[...,0]+xy[...,1]*resolution 

def get_bbox3d_for_blenderobj(camera_transforms, H, W, near=2.0, far=6.0):
    camera_angle_x = float(camera_transforms['camera_angle_x'])
    focal = 0.5*W/np.tan(0.5 * camera_angle_x)

    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    points = []

    for frame in camera_transforms["frames"]:
        c2w = torch.FloatTensor(frame["transform_matrix"])
        rays_o, rays_d = get_rays(directions, c2w)
        
        def find_min_max(pt):
            for i in range(3):
                if(min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if(max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in [0, W-1, H*W-W, H*W-1]:
            min_point = rays_o[i] + near*rays_d[i]
            max_point = rays_o[i] + far*rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)

    return (torch.tensor(min_bound)-torch.tensor([1.0,1.0,1.0]), torch.tensor(max_bound)+torch.tensor([1.0,1.0,1.0]))


def get_bbox3d_for_llff(poses, hwf, near=0.0, far=1.0):
    H, W, focal = hwf
    H, W = int(H), int(W)
    
    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    points = []
    poses = torch.FloatTensor(poses)
    for pose in poses:
        rays_o, rays_d = get_rays(directions, pose)
        rays_o, rays_d = get_ndc_rays(H, W, focal, 1.0, rays_o, rays_d)

        def find_min_max(pt):
            for i in range(3):
                if(min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if(max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in [0, W-1, H*W-W, H*W-1]:
            min_point = rays_o[i] + near*rays_d[i]
            max_point = rays_o[i] + far*rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)

    return (torch.tensor(min_bound)-torch.tensor([0.1,0.1,0.0001]), torch.tensor(max_bound)+torch.tensor([0.1,0.1,0.0001]))


def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    box_min, box_max = bounding_box

    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        pdb.set_trace()
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max-box_min)/resolution
    
    bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()
    voxel_min_vertex = bottom_left_idx*grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0,1.0,1.0])*grid_size

    # hashed_voxel_indices = [] # B x 8 ... 000,001,010,011,100,101,110,111
    # for i in [0, 1]:
    #     for j in [0, 1]:
    #         for k in [0, 1]:
    #             vertex_idx = bottom_left_idx + torch.tensor([i,j,k])
    #             # vertex = bottom_left + torch.tensor([i,j,k])*grid_size
    #             hashed_voxel_indices.append(hash(vertex_idx, log2_hashmap_size))

    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices

def get_plane_vertices_old(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    def box2plane(input):
        in_xy = input[:,:2]#.unsqueeze(1)
        in_xz = input[:,::2]#.unsqueeze(1)
        in_yz = input[:,-2:]#.unsqueeze(1)
        return [in_xy,in_xz,in_yz]

    box_min, box_max = bounding_box

    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        pdb.set_trace()
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max-box_min)/resolution
    
    bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()                   #(B, 3)
    voxel_min_vertex = bottom_left_idx*grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0,1.0,1.0])*grid_size

    # hashed_voxel_indices = [] # B x 8 ... 000,001,010,011,100,101,110,111
    # for i in [0, 1]:
    #     for j in [0, 1]:
    #         for k in [0, 1]:
    #             vertex_idx = bottom_left_idx + torch.tensor([i,j,k])
    #             # vertex = bottom_left + torch.tensor([i,j,k])*grid_size
    #             hashed_voxel_indices.append(hash(vertex_idx, log2_hashmap_size))

    #voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS       #(B, 8, 3)
    #hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)    #(B, 8)

    voxel_indices_xy = bottom_left_idx[:,:2].unsqueeze(1) + SQR_OFFSETS   #(B, 4, 2)
    voxel_indices_xz = bottom_left_idx[:,::2].unsqueeze(1) + SQR_OFFSETS  #(B, 4, 2)
    voxel_indices_yz = bottom_left_idx[:,-2:].unsqueeze(1) + SQR_OFFSETS  #(B, 4, 2)
    hashed_voxel_indices_xy = hash2d(voxel_indices_xy, log2_hashmap_size)    #(B, 4)
    hashed_voxel_indices_xz = hash2d(voxel_indices_xz, log2_hashmap_size)    #(B, 4)
    hashed_voxel_indices_yz = hash2d(voxel_indices_yz, log2_hashmap_size)    #(B, 4)
    hashed_voxel_indices = [hashed_voxel_indices_xy,
                            hashed_voxel_indices_xz,
                            hashed_voxel_indices_yz]
    voxel_min_vertex = box2plane(voxel_min_vertex)
    voxel_max_vertex = box2plane(voxel_max_vertex)
    #pdb.set_trace()

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices

def get_plane_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    def box2plane(input):
        in_xy = input[:,:2]#.unsqueeze(1)
        in_xz = input[:,::2]#.unsqueeze(1)
        in_yz = input[:,-2:]#.unsqueeze(1)
        return [in_xy,in_xz,in_yz]

    box_min, box_max = bounding_box

    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        pdb.set_trace()
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max-box_min)/resolution
    
    bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()                   #(B, 3)
    voxel_min_vertex = bottom_left_idx*grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0,1.0,1.0])*grid_size

    # hashed_voxel_indices = [] # B x 8 ... 000,001,010,011,100,101,110,111
    # for i in [0, 1]:
    #     for j in [0, 1]:
    #         for k in [0, 1]:
    #             vertex_idx = bottom_left_idx + torch.tensor([i,j,k])
    #             # vertex = bottom_left + torch.tensor([i,j,k])*grid_size
    #             hashed_voxel_indices.append(hash(vertex_idx, log2_hashmap_size))

    #voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS       #(B, 8, 3)
    #hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)    #(B, 8)

    voxel_indices_xy = bottom_left_idx[:,:2].unsqueeze(1) + SQR_OFFSETS   #(B, 4, 2)
    voxel_indices_xz = bottom_left_idx[:,::2].unsqueeze(1) + SQR_OFFSETS  #(B, 4, 2)
    voxel_indices_yz = bottom_left_idx[:,-2:].unsqueeze(1) + SQR_OFFSETS  #(B, 4, 2)
    #hashed_voxel_indices_xy = hash2d(voxel_indices_xy, log2_hashmap_size)    #(B, 4)
    #hashed_voxel_indices_xz = hash2d(voxel_indices_xz, log2_hashmap_size)    #(B, 4)
    #hashed_voxel_indices_yz = hash2d(voxel_indices_yz, log2_hashmap_size)    #(B, 4)
    hashed_voxel_indices_xy = xy2index(voxel_indices_xy,resolution)        #(B, 4)
    hashed_voxel_indices_xz = xy2index(voxel_indices_xz,resolution)        #(B, 4)
    hashed_voxel_indices_yz = xy2index(voxel_indices_yz,resolution)        #(B, 4)
    #print(hashed_voxel_indices_yz.shape)
    #pdb.set_trace()
    hashed_voxel_indices = [hashed_voxel_indices_xy,
                            hashed_voxel_indices_xz,
                            hashed_voxel_indices_yz]
    voxel_min_vertex = box2plane(voxel_min_vertex)
    voxel_max_vertex = box2plane(voxel_max_vertex)
    

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices

if __name__=="__main__":
    with open("data/nerf_synthetic/chair/transforms_train.json", "r") as f:
        camera_transforms = json.load(f)
    
    bounding_box = get_bbox3d_for_blenderobj(camera_transforms, 800, 800)
