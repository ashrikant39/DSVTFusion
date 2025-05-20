import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate

def get_points_in_box(points, boxes, box_batch_idxs):
    """
    points -> num_points, 11 (N, 11)
    box -> num_boxes, 11 (M, 11)
    mask_shape -> num_points, num_boxes
    """
    
    batch_idxs = points[:, [0]]
    batch_mask = batch_idxs == box_batch_idxs.T
    
    x, y, z = points[:, [1]], points[:, [2]], points[:, [3]] # N,1
    cx, cy, cz = boxes[:, 0][None], boxes[:, 1][None], boxes[:, 2][None] # (1, M)
    dx, dy, dz, rz = boxes[:, 3][None], boxes[:, 4][None], boxes[:, 5][None], boxes[:, 6][None] # (1, M)
    shift_x, shift_y, shift_z = x - cx, y - cy, z - cz # (N, M)
    
    MARGIN = 1e-1
    cosa, sina = torch.cos(-rz), torch.sin(-rz)
    local_x = shift_x * cosa + shift_y * (-sina)
    local_y = shift_x * sina + shift_y * cosa

    geom_mask = torch.logical_and(abs(shift_z) <= dz / 2.0, 
                          torch.logical_and(abs(local_x) <= dx / 2.0 + MARGIN, 
                                         abs(local_y) <= dy / 2.0 + MARGIN))   
    
    final_in_box_mask = torch.logical_and(geom_mask, batch_mask)
    return final_in_box_mask


# def get_points_in_box_with_class(points, boxes, box_batch_idxs):
#     """
#     points -> num_points, 11 (N, 11)
#     box -> num_boxes, 10 (M, 10)
#     mask_shape -> num_points, num_boxes
#     """
    
#     batch_idxs = points[:, [0]]
#     batch_mask = batch_idxs == box_batch_idxs.T
#     box_label = boxes[:,-1].long() - 1
#     num_points = len(points)
    
#     x, y, z = points[:, [1]], points[:, [2]], points[:, [3]] # N,1
#     cx, cy, cz = boxes[:, 0][None], boxes[:, 1][None], boxes[:, 2][None] # (1, M)
#     dx, dy, dz, rz = boxes[:, 3][None], boxes[:, 4][None], boxes[:, 5][None], boxes[:, 6][None] # (1, M)
#     shift_x, shift_y, shift_z = x - cx, y - cy, z - cz # (N, M)
    
#     MARGIN = 1e-1
#     cosa, sina = torch.cos(-rz), torch.sin(-rz)
#     local_x = shift_x * cosa + shift_y * (-sina)
#     local_y = shift_x * sina + shift_y * cosa

#     geom_mask = torch.logical_and(abs(shift_z) <= dz / 2.0, 
#                           torch.logical_and(abs(local_x) <= dx / 2.0 + MARGIN, 
#                                          abs(local_y) <= dy / 2.0 + MARGIN))   
    
#     final_in_box_mask = torch.logical_and(geom_mask, batch_mask)        
#     return final_in_box_mask


def encode_bbox(bboxes):
    
    # [x, y, z, w, l, h, yaw, vx, vy, class_id]
    targets = torch.zeros([bboxes.shape[0], 10]).to(bboxes)
    targets[:, :3] = bboxes[:, :3] # xyz
    targets[:, 3:6] = bboxes[:, 3:6].log() # lwh
    targets[:, 6] = torch.sin(bboxes[:, 6]) # sin yaw
    targets[:, 7] = torch.cos(bboxes[:, 6]) # cos yaw
    targets[:, 8:10] = bboxes[:, 7:9] # vx, vy
    
    return targets


class PFNLayerV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):

        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated


class DynamicPillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor((points[:, [1,2]] - self.point_cloud_range[[0,1]]) / self.voxel_size[[0,1]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0,1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        # points -> batch_idx, x, y, z, i, e
        # points_coords -> x_idx, y_idx
        # Batch_idx * (H*W) + x_idx * grid_size + y_idx
        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        # uniq_inv -> A mapping on how to combine points belonging to the same pillar
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                    (unq_coords % self.scale_xy) // self.scale_y,
                                    unq_coords % self.scale_y,
                                    torch.zeros(unq_coords.shape[0]).to(unq_coords.device).int()
                                    ), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        batch_dict['pillar_features'] = batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = voxel_coords
        batch_dict['voxel_size'] = self.voxel_size

        return batch_dict


class DynamicPillarWithBoxVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]
    

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor((points[:, [1,2]] - self.point_cloud_range[[0,1]]) / self.voxel_size[[0,1]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0,1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()
        
        # points -> batch_idx, x, y, z, i, e
        # points_coords -> x_idx, y_idx
        # Batch_idx * (H*W) + x_idx * grid_size + y_idx
        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        # uniq_inv -> A mapping on how to combine points belonging to the same pillar
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                    (unq_coords % self.scale_xy) // self.scale_y,
                                    unq_coords % self.scale_y,
                                    torch.zeros(unq_coords.shape[0]).to(unq_coords.device).int()
                                    ), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]                    
        # box_centers = batch_dict['box_centers'] # (batch, boxes_per_batch, box_dim)
        gt_boxes = batch_dict['gt_boxes']
        box_centers = gt_boxes[...,:3]
        
        # clamp to belong to pc range
        box_centers = torch.clamp(
            box_centers, 
            min=self.point_cloud_range[0:3],
            max=self.point_cloud_range[3:6]
        )
        
        min_shifted_box_centers = box_centers[:, :, [0,1]] - self.point_cloud_range[[0,1]]
        normalized_coords = -1.0 + 2*min_shifted_box_centers/(self.point_cloud_range[[3,4]] - self.point_cloud_range[[0,1]])
                    
        box_center_voxel_coords = torch.floor(min_shifted_box_centers / self.voxel_size[[0,1]]).long() 
        
        batch_size, seq_len, _ = box_center_voxel_coords.shape
        batch_idxs = torch.arange(batch_size, device=box_center_voxel_coords.device).view(batch_size, 1, 1).expand(-1, seq_len, 1)
        z_coords = torch.zeros_like(batch_idxs, device=box_center_voxel_coords.device)
                    
        box_center_voxel_coords = torch.cat([batch_idxs, z_coords, box_center_voxel_coords], dim=-1).contiguous()
        box_center_voxel_coords = box_center_voxel_coords.view(-1, 4)
        
        batch_dict['box_coords'] = box_center_voxel_coords
                
        bev_feats = batch_dict['bev_features']
        _, C, _, _ = bev_feats.shape
        
        box_feats = F.grid_sample(bev_feats, normalized_coords[:,:,None,:], align_corners=True).permute(0,2,1,3).squeeze().contiguous()
        
        batch_dict['box_features'] = box_feats.reshape(-1, C)
        batch_dict['boxes_per_batch'] = seq_len
        
        batch_dict['pillar_features'] = batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = voxel_coords
        batch_dict['voxel_size'] = self.voxel_size

        return batch_dict



class DynamicPillarWithClassFeatsVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()
        
        self.class_feats = torch.load("work_dirs/class_features_sweep1.pth", map_location='cuda')['class_feature']
        self.class_feats.requires_grad = False

    def get_output_feature_dim(self):
        return self.num_filters[-1]
    

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor((points[:, [1,2]] - self.point_cloud_range[[0,1]]) / self.voxel_size[[0,1]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0,1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()
        
        # points -> batch_idx, x, y, z, i, e
        # points_coords -> x_idx, y_idx
        # Batch_idx * (H*W) + x_idx * grid_size + y_idx
        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        # uniq_inv -> A mapping on how to combine points belonging to the same pillar
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                    (unq_coords % self.scale_xy) // self.scale_y,
                                    unq_coords % self.scale_y,
                                    torch.zeros(unq_coords.shape[0]).to(unq_coords.device).int()
                                    ), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]                    
        # box_centers = batch_dict['box_centers'] # (batch, boxes_per_batch, box_dim)
        gt_boxes = batch_dict['gt_boxes']
        batch_size, num_boxes, _ = gt_boxes.shape
    
        batch_idxs = torch.arange(batch_size, device=voxel_coords.device).view(batch_size, 1, 1).expand(-1, num_boxes, 1).reshape(-1, 1)
        
        class_labels = (gt_boxes[...,-1].long() - 1).reshape(-1)
            
        box_centers = gt_boxes[...,:3].reshape(-1, 3)
        box_center_filter =  box_centers.sum(dim=-1) != 0
        box_centers = box_centers[box_center_filter]
        class_labels = class_labels[box_center_filter]
        batch_idxs = batch_idxs[box_center_filter]
        
        # clamp to belong to pc range
        box_centers = torch.clamp(
            box_centers, 
            min=self.point_cloud_range[0:3],
            max=self.point_cloud_range[3:6]
        )
        
        min_shifted_box_centers = box_centers[:, [0,1]] - self.point_cloud_range[[0,1]]
        # normalized_coords = -1.0 + 2*min_shifted_box_centers/(self.point_cloud_range[[3,4]] - self.point_cloud_range[[0,1]])
                    
        box_center_voxel_coords = torch.floor(min_shifted_box_centers / self.voxel_size[[0,1]]).long() 
        
        # batch_idxs = torch.arange(batch_size, device=box_center_voxel_coords.device).view(batch_size, 1, 1).expand(-1, seq_len, 1)
        z_coords = torch.zeros_like(batch_idxs, device=box_center_voxel_coords.device)
        
        box_center_voxel_coords = torch.cat([batch_idxs, z_coords, box_center_voxel_coords], dim=-1).contiguous()
                
        batch_dict['box_coords'] = box_center_voxel_coords          
        batch_dict['box_features'] = self.class_feats[class_labels]
        batch_dict['boxes_per_batch'] = num_boxes
        
        batch_dict['pillar_features'] = batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = voxel_coords
        batch_dict['voxel_size'] = self.voxel_size

        return batch_dict


class DynamicPillarWithFeatureSeg(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        self.token_dim = self.model_cfg.TOKEN_DIM
        if self.with_distance:
            num_point_features += 1

        num_point_features += self.token_dim
        
        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()
            
        self.mask_token = nn.Parameter(torch.randn(self.token_dim,), requires_grad=True)
        self.null_token = nn.Parameter(torch.randn(self.token_dim,), requires_grad=True)
        

    def get_output_feature_dim(self):
        return self.num_filters[-1]
    

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor((points[:, [1,2]] - self.point_cloud_range[[0,1]]) / self.voxel_size[[0,1]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0,1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()
        
        # points -> batch_idx, x, y, z, i, e
        # points_coords -> x_idx, y_idx
        # Batch_idx * (H*W) + x_idx * grid_size + y_idx
        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        # uniq_inv -> A mapping on how to combine points belonging to the same pillar
        
        gt_boxes = batch_dict['gt_boxes']
        batch_size, seq_len, _ = gt_boxes.shape
        
        seg_feature = torch.repeat_interleave(self.null_token[None], len(features), dim=0)
        
        
        if 0 not in gt_boxes.shape:
            gt_boxes = gt_boxes.reshape(batch_size*seq_len, -1)
            
            invalid_box_mask = (gt_boxes.sum(dim=-1) != 0)
            gt_boxes = gt_boxes[invalid_box_mask]
            
            batch_idxs = torch.arange(batch_size, device=points.device).view(batch_size, 1, 1).expand(-1, seq_len, 1).reshape(-1,1)
            valid_box_batch_idxs = batch_idxs[invalid_box_mask]
            
            points_in_box_mask = get_points_in_box(points, gt_boxes, valid_box_batch_idxs)
            any_point_in_box_mask = points_in_box_mask.any(dim=1)
            seg_feature[any_point_in_box_mask] = self.mask_token
        
        features = torch.cat([features, seg_feature], dim=1)
                
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                    (unq_coords % self.scale_xy) // self.scale_y,
                                    unq_coords % self.scale_y,
                                    torch.zeros(unq_coords.shape[0]).to(unq_coords.device).int()
                                    ), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]                    
                    
        batch_dict['pillar_features'] = batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = voxel_coords
        batch_dict['voxel_size'] = self.voxel_size
        
        return batch_dict


class DynamicPillarWithClassSeg(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        self.token_dim = self.model_cfg.TOKEN_DIM
        self.num_classes = self.model_cfg.NUM_CLASSES
        
        if self.with_distance:
            num_point_features += 1

        num_point_features += self.token_dim
        
        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()
            
        self.mask_token = nn.Parameter(torch.randn(self.num_classes + 1, self.token_dim,), requires_grad=False)        

    def get_output_feature_dim(self):
        return self.num_filters[-1]
    

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor((points[:, [1,2]] - self.point_cloud_range[[0,1]]) / self.voxel_size[[0,1]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0,1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()
        
        # points -> batch_idx, x, y, z, i, e
        # points_coords -> x_idx, y_idx
        # Batch_idx * (H*W) + x_idx * grid_size + y_idx
        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        # uniq_inv -> A mapping on how to combine points belonging to the same pillar
        
        gt_boxes = batch_dict['gt_boxes']
        batch_size, seq_len, _ = gt_boxes.shape
        num_points = len(points)
        
        
        if 0 in gt_boxes.shape:
            seg_feature = torch.repeat_interleave(self.mask_token[[-1]], len(features), dim=0)
        
        else:
            gt_boxes = gt_boxes.reshape(batch_size*seq_len, -1)
            
            valid_box_mask = (gt_boxes.sum(dim=-1) != 0)
            gt_boxes = gt_boxes[valid_box_mask]
            box_label = gt_boxes[:,-1].long() - 1
                        
            batch_idxs = torch.arange(batch_size, device=points.device).view(batch_size, 1, 1).expand(-1, seq_len, 1).reshape(-1,1)
            valid_box_batch_idxs = batch_idxs[valid_box_mask]
            

            final_in_box_mask = get_points_in_box(points, gt_boxes, valid_box_batch_idxs, self.num_classes)
            point_idxs, box_idxs = torch.where(final_in_box_mask)  
            
            class_labels_to_points = torch.full((num_points,), fill_value=self.num_classes, device=points.device, dtype=torch.long)
            class_labels_to_points[point_idxs] = box_label[box_idxs]
            seg_feature = self.mask_token[class_labels_to_points]
        
        features = torch.cat([features, seg_feature], dim=1)
                
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                    (unq_coords % self.scale_xy) // self.scale_y,
                                    unq_coords % self.scale_y,
                                    torch.zeros(unq_coords.shape[0]).to(unq_coords.device).int()
                                    ), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]                    
                    
        batch_dict['pillar_features'] = batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = voxel_coords
        batch_dict['voxel_size'] = self.voxel_size
        
        return batch_dict
    

class DynamicPillarWithFullBoxSeg(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        self.token_dim = self.model_cfg.TOKEN_DIM
        self.num_classes = self.model_cfg.NUM_CLASSES
        
        if self.with_distance:
            num_point_features += 1

        
        num_point_features += self.token_dim # for class label
        num_point_features += 10 # for box info
        
        
        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()
        
        self.box_null_token = nn.Parameter(torch.randn(10), requires_grad=False)
        self.mask_token = nn.Parameter(torch.randn(self.num_classes + 1, self.token_dim,), requires_grad=False)     

    def get_output_feature_dim(self):
        return self.num_filters[-1]
    

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor((points[:, [1,2]] - self.point_cloud_range[[0,1]]) / self.voxel_size[[0,1]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0,1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()
        
        # points -> batch_idx, x, y, z, i, e
        # points_coords -> x_idx, y_idx
        # Batch_idx * (H*W) + x_idx * grid_size + y_idx
        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        # uniq_inv -> A mapping on how to combine points belonging to the same pillar
        
        gt_boxes = batch_dict['gt_boxes']
        batch_size, seq_len, _ = gt_boxes.shape
        
        full_feature = torch.repeat_interleave(torch.cat([self.box_null_token, self.mask_token[-1]], dim=0)[None, :], repeats=len(features), dim=0).to(gt_boxes)

        if 0 not in gt_boxes.shape:
    
            gt_boxes = gt_boxes.reshape(batch_size*seq_len, -1)
                                                            
            valid_box_mask = (gt_boxes.sum(dim=-1) != 0)
            gt_boxes = gt_boxes[valid_box_mask]
                        
            batch_idxs = torch.arange(batch_size, device=points.device).view(batch_size, 1, 1).expand(-1, seq_len, 1).reshape(-1,1)
            valid_box_batch_idxs = batch_idxs[valid_box_mask]
                     
            final_in_box_mask = get_points_in_box(points, gt_boxes, valid_box_batch_idxs)
            point_idxs, box_idxs = torch.where(final_in_box_mask)
            
            class_labels = gt_boxes[box_idxs][:,-1].long() - 1
            
            encoded_boxes = encode_bbox(gt_boxes)
            
            full_feature[point_idxs][:, :10] = encoded_boxes[box_idxs]
            full_feature[point_idxs][:, 10:] = self.mask_token[class_labels]
        
        #                   (N_full, 11)
        features = torch.cat([features, full_feature], dim=1)
                
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                    (unq_coords % self.scale_xy) // self.scale_y,
                                    unq_coords % self.scale_y,
                                    torch.zeros(unq_coords.shape[0]).to(unq_coords.device).int()
                                    ), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]                    
                    
        batch_dict['pillar_features'] = batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = voxel_coords
        batch_dict['voxel_size'] = self.voxel_size
        
        return batch_dict


class DynamicPillarVFE_3d(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor((points[:, [1,2,3]] - self.point_cloud_range[[0,1,2]]) / self.voxel_size[[0,1,2]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0,1,2]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xyz + \
                       points_coords[:, 0] * self.scale_yz + \
                       points_coords[:, 1] * self.scale_z + \
                       points_coords[:, 2]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        # f_center[:, 2] = points_xyz[:, 2] - self.z_offset
        f_center[:, 2] = points_xyz[:, 2] - (points_coords[:, 2].to(points_xyz.dtype) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        batch_dict['pillar_features'] = batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = voxel_coords

        return batch_dict
