from .detector3d_template import Detector3DTemplate
import torch, torch.nn.functional as F
import torch.nn as nn
import os
from ...utils.spconv_utils import find_all_spconv_keys
import math
from einops import rearrange
from ..dense_heads.target_assigner.hungarian_assigner import HungarianAssigner3D
import pdb
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_and_union_gpu
from pcdet.utils.box_utils import boxes_to_corners_3d
from pcdet.utils.loss_utils import SigmoidFocalClassificationLoss


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce)
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()



def encode_bbox(bboxes, pcd_range):
    
    # [x, y, z, w, l, h, yaw, vx, vy, class_id]
    # output_dir
    # [x, y, z, logw, logl, logh, sin yaw, cos yaw, vx, vy]
    box_dims = bboxes.shape
    targets = torch.zeros([*box_dims[:-1], 10]).to(bboxes)

    targets[..., :3] = (bboxes[..., :3] - pcd_range[:3])/(pcd_range[3:] - pcd_range[:3]) # xyz
    targets[..., 3:6] = bboxes[..., 3:6].log() # lwh
    targets[..., 6] = torch.sin(bboxes[..., 6]) # sin yaw
    targets[..., 7] = torch.cos(bboxes[..., 6]) # cos yaw
    targets[..., 8:10] = bboxes[..., 7:9] # vx, vy
    
    return targets



def decode_bbox(bboxes, pcd_range):
    
    # [x, y, z, log w, log l, log h, sin yaw, cos yaw, vx, vy]
    
    xyz = bboxes[..., :3] * (pcd_range[3:] - pcd_range[:3]) + pcd_range[:3] # xyz
    lwh = bboxes[..., 3:6].exp() # lwh
    yaw = torch.atan2(bboxes[..., 6:7], bboxes[..., 7:8]) # sin yaw
    velocity = bboxes[..., 8:10] # vx, vy    

    return torch.cat([xyz, lwh, yaw, velocity], dim=-1)



def calculate_generalized_iou3d(pred_boxes, gt_boxes):
    """
    boxes of shape (N, 7)
    """
    iou, union = boxes_iou3d_and_union_gpu(pred_boxes, gt_boxes)

    pred_corners = boxes_to_corners_3d(pred_boxes)
    gt_corners = boxes_to_corners_3d(gt_boxes)
    corners = torch.cat([pred_corners, gt_corners], dim=1)

    enclosing_min = corners.min(dim=1).values
    enclosing_max = corners.max(dim=1).values

    dims = enclosing_min - enclosing_max
    enclosing_vol = torch.prod(dims, dim=1)
    
    return iou - (enclosing_vol - union)/enclosing_vol


def sinusoidal_time_embedding(boxes:torch.Tensor) -> torch.Tensor:
    """
    Adds sinusoidal time embeddings to a tensor of shape (B, T, N, D)
    
    Args:
        x: Input tensor of shape (B, T, N, D), where
           B = batch size
           T = number of time steps
           N = number of spatial tokens per time step
           D = feature dimension (must be even)
           
    Returns:
        Tensor with sinusoidal time embeddings added to the input.
    """
    _, T, _, D = boxes.shape 
    assert D % 2 == 0, "Feature dimension must be even for sinusoidal embeddings."

    # Create time positions [0, 1, ..., T]
    position = torch.arange(T+1).unsqueeze(1).to(boxes)  # (T + 1, 1)
    div_term = torch.exp(torch.arange(0, D, 2).to(boxes) * (-math.log(10000.0) / D))  # (D/2,)

    pe = torch.zeros(T + 1, D).to(boxes)  # (T + 1, D)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # Expand to match input shape: (1, T + 1, 1, D):
    pe = pe.unsqueeze(0).unsqueeze(2)
    
    return pe



# def get_voxel_centers_in_boxes(gt_boxes, voxel_size):
#     """
#     Args:
#         gt_boxes: Tensor of shape (B, N, 10) → [x, y, z, l, w, h, yaw, vx, vy, label]
#         voxel_size: Tensor of shape (3,) → [vx, vy, vz]

#     Returns:
#         voxel_centers: (B, N, M, 3) tensor of voxel center coordinates in world frame
#         valid_mask: (B, N, M) boolean mask indicating which of the M positions are valid per box
#     """
#     B, N, _ = gt_boxes.shape
#     device = gt_boxes.device
#     voxel_size = voxel_size.to(device)

#     # 1. Extract box params
#     center = gt_boxes[..., 0:3]  # (B, N, 3)
#     dims   = gt_boxes[..., 3:6]  # (B, N, 3)
#     yaws   = gt_boxes[..., 6]    # (B, N)

#     # 2. Get voxel counts (number of voxels along each axis per box)
#     n_voxels = torch.clamp((dims / voxel_size).floor().int(), min=1)  # (B, N, 3)
#     max_n = n_voxels.amax(dim=(0,1))  # (3,) → max grid size along each axis

#     # 3. Create a full grid of voxel indices
#     dz = torch.arange(0, max_n[2], 8, device=device)
#     dy = torch.arange(0, max_n[1], 8, device=device)
#     dx = torch.arange(0, max_n[0], 8, device=device)
    
    
#     grid_z, grid_y, grid_x = torch.meshgrid(dz, dy, dx, indexing='ij')
    
#     grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).float()  # (Z, Y, X, 3)
#     G = grid.numel() // 3
#     grid = grid.view(G, 3)  # (G, 3)

#     # 4. Center the grid and scale to voxel centers
#     grid = grid * voxel_size + voxel_size / 2.0  # (G, 3)

#     # 5. Create expanded tensors to match (B, N, G, 3)
#     grid = grid.view(1, 1, G, 3).expand(B, N, G, 3)  # (B, N, G, 3)
#     dims = dims.unsqueeze(2)  # (B, N, 1, 3)
#     center = center.unsqueeze(2)  # (B, N, 1, 3)
#     yaw = yaws.unsqueeze(2)  # (B, N, 1)

#     # 6. Compute local voxel positions (relative to box center)
#     local = grid - dims / 2  # center at origin per box

#     # 7. Rotation matrix for each box
#     cos_yaw = torch.cos(yaw)
#     sin_yaw = torch.sin(yaw)
#     rot = torch.zeros(B, N, 3, 3, device=device)
#     rot[..., 0, 0] = cos_yaw.squeeze(-1)
#     rot[..., 0, 1] = -sin_yaw.squeeze(-1)
#     rot[..., 1, 0] = sin_yaw.squeeze(-1)
#     rot[..., 1, 1] = cos_yaw.squeeze(-1)
#     rot[..., 2, 2] = 1
    
#     # 8. Rotate local grid (B,N,G,3) x (B,N,3,3)
#     rotated = torch.matmul(local, rot)  # (B, N, G, 3)

#     # 9. Translate to world frame
#     voxel_centers = rotated + center  # (B, N, G, 3)

#     # 10. Mask out-of-bound voxels
#     in_bound = (grid <= dims).all(dim=-1)  # (B, N, G)

#     return voxel_centers, in_bound  # (B,N,G,3), (B,N,G)


class TemporalBoxDecoder(nn.Module):

    def __init__(self, box_dim, feature_dim, n_heads, n_layers, n_embeddings, n_classes) -> None:
        super().__init__()
        

        self.box_to_feature = nn.Sequential(
                nn.Linear(in_features=box_dim, out_features=feature_dim),
                nn.ReLU(inplace=True)
                )
        
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=feature_dim, nhead=n_heads, dim_feedforward=feature_dim, batch_first=True), num_layers=n_layers)
        self.queries = nn.Embedding(n_embeddings, feature_dim)
        self.class_embedding = nn.Embedding(n_classes, feature_dim)
        self.feature_to_box = nn.Linear(in_features=feature_dim, out_features=box_dim)
        self.box_score = nn.Linear(in_features=box_dim, out_features=1)
        self.classifier = nn.Linear(in_features=feature_dim, out_features=n_classes)

        self.box_dim = box_dim
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_queries = n_embeddings


    def forward(self, boxes:torch.Tensor, valid_box_mask:torch.Tensor, labels:torch.Tensor):
        """
        Given pure boxes from previous timesteps, this model is to learn a transformation for the box.
        Each box from t_a to T are treated independently (there is no temporal relation between boxes)
        The prediction happens from t_a to T.


        Make use of learnt queries to get proposed regions to perform point labeling.
        """
        # boxes -> (batch_size, num_timesteps, num_boxes, box_dim)
        batch_size, num_timesteps, num_boxes, box_dim  = boxes.shape
        box_feats = torch.zeros((batch_size * num_timesteps * num_boxes, self.feature_dim)).to(boxes)

        boxes = rearrange(boxes, 'b t n f -> (b t n) f')

        key_padding_mask = rearrange(~valid_box_mask, 'b t n -> b (t n)')

        box_feats[valid_box_mask.flatten()] = self.box_to_feature(boxes[valid_box_mask.flatten()]) + self.class_embedding(labels[valid_box_mask])
        box_feats = box_feats.reshape((batch_size, num_timesteps, num_boxes, self.feature_dim))
        
        te = sinusoidal_time_embedding(box_feats)

        curr_te, prev_te = te[:, 0, ...], te[:, 1:, ...]
        box_feats += prev_te

        box_feats = rearrange(box_feats, 'b t n f -> b (t n) f')
        queries = self.queries.weight.unsqueeze(0).expand(batch_size, -1, -1) + curr_te

        query_feats = self.decoder(tgt=queries, memory=box_feats, memory_key_padding_mask=key_padding_mask)
        pred_boxes = self.feature_to_box(query_feats)
        class_scores = self.classifier(query_feats)
        box_scores = self.box_score(pred_boxes)

        return pred_boxes, box_scores, class_scores


class TransFusion(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self,batch_dict):
        disp_dict = {}

        loss_trans, tb_dict = batch_dict['loss'],batch_dict['tb_dict']
        tb_dict = {
            'loss_trans': loss_trans.item(),
            **tb_dict
        }

        loss = loss_trans
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
    
    
    
class TransfusionWrapper(nn.Module):
    
    def __init__(self, model_cfg, num_class, dataset):
        
        super().__init__()
        
        ref_model_cfg = model_cfg['REF_MODEL']
        model_cfg = model_cfg['TRAIN_MODEL']
        
        self.ref_model = TransFusion(ref_model_cfg, num_class, dataset)
        self.model = TransFusion(model_cfg, num_class, dataset)
        
        self.ref_model.module_list = self.ref_model.build_networks()
        self.model.module_list = self.model.build_networks()
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        # for name, param in self.model.named_parameters():
        #     if 'cross_attention_layer' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        
        
    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state
    
    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1
        
    
    def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        if not pre_trained_path is None:
            pretrain_checkpoint = torch.load(pre_trained_path, map_location=loc_type)
            pretrain_model_state_disk = pretrain_checkpoint['model_state']
            model_state_disk.update(pretrain_model_state_disk)

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch
        
        
    def get_training_loss(self,batch_dict):
        disp_dict = {}

        loss_trans, tb_dict = batch_dict['loss'],batch_dict['tb_dict']
        tb_dict = {
            'loss_trans': loss_trans.item(),
            **tb_dict
        }

        loss = loss_trans
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.model.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
    
        
    def forward(self, batch_dict):
                
        with torch.no_grad():
            self.ref_model.eval()
            
            no_grad_dict = batch_dict.copy()
            for cur_module in self.ref_model.module_list:
                no_grad_dict = cur_module(no_grad_dict)
        
        gt_boxes = batch_dict['gt_boxes']    
        
        if 0 not in gt_boxes.shape or self.training:    
            batch_dict['bev_features'] = no_grad_dict['spatial_features']
            
            for cur_module in self.model.module_list:
                batch_dict = cur_module(batch_dict)
                
        else:
            batch_dict = no_grad_dict.copy()
            
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
        

class TransFusionTemporalFullSweepBoxModel(TransFusion):    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for module in self.module_list:
            for param in module.parameters():
                param.requires_grad = False
        
        self.num_classes = kwargs.get('num_classes', 10)

        self.temporal_box_model = TemporalBoxDecoder(
                box_dim = kwargs.get('box_dim', 10),
                feature_dim = kwargs.get('box_decoder_feature_dim', 256),
                n_heads = kwargs.get('n_box_decoder_heads', 8),
                n_layers = kwargs.get('n_box_decoder_layers', 6),
                n_embeddings = kwargs.get('n_box_decoder_embeds', 200),
                n_classes = kwargs.get('num_classes', 10)
            )

        self.pcd_range = self.vfe.point_cloud_range
        self.box_code_length = 10

        default_assigner = {
                "cls_cost": {"weight": 0.0,},
                "reg_cost": {"weight": 1.0,},
                "iou_cost": {"weight": 1.0,},
                }
        
        self.reg_loss_weight = kwargs.get('reg_loss_weight', 1.0)
        self.score_loss_weight = kwargs.get('reg_loss_weight', 0.5)
        self.cls_loss_weight = kwargs.get('class_loss_weight', 1.0)

        self.cls_loss = SigmoidFocalClassificationLoss()

        self.bbox_assigner = HungarianAssigner3D(**kwargs.get('assigner', default_assigner))
        self.init_weights()


    def init_weights(self):

        for module in self.temporal_box_model.modules():

            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.kaiming_uniform_(module.weight)

        
    def process_boxes(self, box_dict, keys, use_score = False):
        
        num_boxes = []
        batch_size, _, box_dim = box_dict[keys[0]].shape
        num_timesteps = len(keys)
        # for idx, box_dict in enumerate(box_dicts):
            
        #     scores = box_dict['pred_scores']
        #     thresh_mask = scores > 0.5 * max(scores)
        #     num_boxes.append(thresh_mask.sum().item())
            
        #     box_dicts[idx]['pred_scores'] = box_dict['pred_scores'][thresh_mask]
        #     box_dicts[idx]['pred_boxes'] = box_dict['pred_boxes'][thresh_mask]
        #     box_dicts[idx]['pred_labels'] = box_dict['pred_labels'][thresh_mask]
        
        # num_max_boxes = max(num_boxes)
        # processed_boxes = torch.zeros((batch_size, num_max_boxes, 10)).to(box_dicts[0]['pred_boxes'])
        
        # for idx, curr_num_boxes in enumerate(num_boxes):
        #     processed_boxes[idx][:curr_num_boxes] = torch.cat([box_dicts[idx]['pred_boxes'], box_dicts[idx]['pred_labels'][:, None]], dim=1)
        

        for idx, key in enumerate(keys):

            boxes = box_dict[key]
            num_boxes.append(boxes.shape[1])

        total_boxes = max(num_boxes)
        processed_boxes = torch.zeros((batch_size, num_timesteps, total_boxes, box_dim)).to(box_dict[keys[0]])
        
        for idx, key in enumerate(keys):
            
            boxes_taken = num_boxes[idx]
            processed_boxes[:, idx, :boxes_taken, :] = box_dict[key]

        return processed_boxes

    
    def predict(self, boxes: torch.Tensor):
        
        B, T, N, D = boxes.shape
        
        valid_box_mask = boxes.sum(dim=-1) != 0
        encoded_boxes = boxes.new_zeros((B, T, N, self.box_code_length))
        labels = torch.zeros((B, T, N), dtype=torch.long, device=boxes.device)
        labels[valid_box_mask] = boxes[valid_box_mask][:, -1].long() - 1
        encoded_boxes[valid_box_mask] = encode_bbox(boxes[valid_box_mask], self.pcd_range)
        
        return self.temporal_box_model(encoded_boxes, valid_box_mask, labels)

    
    def train_box_decoder(self, boxes: torch.Tensor, gt_boxes: torch.Tensor):

        """

        boxes: boxes from previous timesteps
        gt_boxes: GT boxes at current timesteps

        NOTE: We need decoded boxes for Hungarian Matching and encoded box for loss computation
        
        For matching -> eg. pred-> (200, 9), gt-> (29, 9)

        get_targets_single : line 308
        
        NOTE: After Hungarian Assignment, only positive boxes have a regression loss.
        """
        gt_box_tensor, gt_box_labels = gt_boxes[...,:-1], gt_boxes[...,-1]
        gt_box_labels = gt_box_labels.long() - 1
        
        pred_boxes, box_scores, class_scores = self.predict(boxes) # class_scores -> (batch, 200, 10)      
        batch_size, num_preds, box_dim = pred_boxes.shape

        decoded_pred_boxes = decode_bbox(pred_boxes, self.pcd_range)
        
        valid_gt_mask = gt_boxes.sum(dim=-1) != 0
        total_boxes = valid_gt_mask.sum()
        
        target_boxes = gt_box_tensor.new_zeros((batch_size, num_preds, box_dim))
        target_box_weights = torch.zeros_like(target_boxes)
        target_scores = torch.zeros_like(box_scores)
        
        target_labels = gt_box_labels.new_zeros((batch_size, num_preds, self.num_classes))
        label_weights = gt_box_labels.new_zeros((batch_size, num_preds))
        # gIou3d = 0.0

        for idx in range(batch_size):

            valid_gt_tensor = gt_box_tensor[idx][valid_gt_mask[idx]]
            valid_gt_labels = gt_box_labels[idx][valid_gt_mask[idx]]

            assigned_gt_inds, ious = self.bbox_assigner.assign(
                    decoded_pred_boxes.detach()[idx],
                    valid_gt_tensor,
                    valid_gt_labels,
                    class_scores[idx : idx + 1].permute(0,2,1),
                    self.pcd_range)
    
            pos_inds_locs = torch.nonzero(assigned_gt_inds, as_tuple=True)[0].unique()
            pos_gt_inds = assigned_gt_inds[pos_inds_locs] - 1
            
            gt_box_ordered = valid_gt_tensor[pos_gt_inds]
            target_boxes[idx, pos_inds_locs] = encode_bbox(gt_box_ordered, self.pcd_range)
            target_box_weights[idx, pos_inds_locs] = 1.0
            target_scores[idx, pos_inds_locs] = 1.0

            target_labels[idx, pos_inds_locs] = F.one_hot(valid_gt_labels[pos_gt_inds], self.num_classes)
            label_weights[idx, pos_inds_locs] = 1.0

        # gIou3d += calculate_generalized_iou3d(decoded_pred_boxes[idx, :, :7], gt_box_ordered[:, :7])
        loss_dict = dict()
        reg_loss =  (F.l1_loss(pred_boxes, target_boxes, reduction='none') * target_box_weights).sum() / total_boxes 
        loss_dict['reg_loss'] = reg_loss.item()
        # loss_dict['GIoU3D_loss'] = 1.0 - gIou3d/batch_size

        score_loss = focal_loss(box_scores, target_scores)
        loss_dict['score_loss'] = score_loss.item()

        cls_loss = self.cls_loss(class_scores, target_labels, label_weights).sum()/max(total_boxes, 1)
        
        loss_dict['cls_loss'] = cls_loss.item()
        loss = reg_loss * self.reg_loss_weight + score_loss * self.score_loss_weight + self.cls_loss_weight * cls_loss
        
        return loss, loss_dict


    def forward(self, batch_dict):
        
        """
        Check box shapes for all timesteps.
        """

        gt_boxes = batch_dict['gt_boxes']
        batch_size = batch_dict['batch_size']

        prev_keys = [key for key in batch_dict.keys() if 'boxes_' in key]

        if self.training:

            prev_boxes_padded = self.process_boxes(batch_dict, prev_keys)
            loss, loss_dict = self.train_box_decoder(prev_boxes_padded, gt_boxes)

            ret_dict = {
                "loss": loss
            }

            tb_dict = dict()

            return ret_dict, tb_dict, loss_dict

        else:
            
            no_prev_data_idxs = []
            prev_data_present_idxs = []

            recall_dict = dict()

            for idx in range(batch_size):
                if batch_dict['prev_data_present'][idx] is False:
                    no_prev_data_idxs.append(idx)
                else:
                    prev_data_present_idxs.append(idx)

            prev_boxes_padded = self.process_boxes(batch_dict, prev_keys)[prev_data_present_idxs]
            pred_dicts = []

            pred_boxes, box_scores, class_scores = self.predict(prev_boxes_padded)
            decoded_boxes = decode_bbox(pred_boxes, self.pcd_range)
            box_scores = box_scores.sigmoid()
            class_labels = class_scores.max(dim=-1).indices

            box_scores.sigmoid()
            score_mask = box_scores > 0.2

            for idx in range(batch_size):
                
                pred = dict()

                if idx in prev_data_present_idxs:
                    pred["pred_scores"] = box_scores[idx][score_mask[idx]]
                    pred["pred_boxes"] = decoded_boxes[idx][score_mask[idx][:, 0]]
                    pred["pred_labels"] = class_labels[idx][score_mask[idx][:, 0]]

                    pred_dicts.append(pred)
            
                else:
                    pred['pred_scores'] = gt_boxes.new_zeros(self.temporal_box_model.n_queries)
                    pred['pred_boxes'] = gt_boxes.zeros((self.temporal_box_model.n_queries, 9))
                    pred['pred_labels'] = torch.zeros(self.temporal_box_model.n_queries, dtype=torch.long, device = gt_boxes.device)
                
                    pred_dicts.append(pred)
            
            return pred_dicts, recall_dict
        # prev_batch_dict = batch_dict.copy()
        # prev_batch_dict['points'] = batch_dict['prev_points']
        
        # gt_boxes = batch_dict['gt_boxes']
        # prev_batch_dict['prev_boxes'] = torch.empty((0, 0, 10)).to(gt_boxes)
        
        # with torch.no_grad():        
        #     for cur_module in self.module_list:
        #         cur_module.training = False
        #         prev_batch_dict = cur_module(prev_batch_dict)
            
        #     box_dicts = prev_batch_dict['final_box_dicts']
        #     batch_dict['prev_boxes'] = self.process_boxes(box_dicts)
        
        # for cur_module in self.module_list:
            
        #     if self.training:
        #         cur_module.training = True
                
        #     batch_dict = cur_module(batch_dict)
        
        
        # if self.training:
        #     loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

        #     ret_dict = {
        #         'loss': loss
        #     }
        #     return ret_dict, tb_dict, disp_dict
        # else:
        #     pred_dicts, recall_dicts = self.post_processing(batch_dict)
        #     return pred_dicts, recall_dicts
    
    
# class TransfusionTemporalWrapper(nn.Module):
    
#     def __init__(self, model_cfg, num_class, dataset):
        
#         super().__init__()
        
#         ref_model_cfg = model_cfg['REF_MODEL']
#         model_cfg = model_cfg['TRAIN_MODEL']
        
#         self.ref_model = TransFusion(ref_model_cfg, num_class, dataset)
#         self.model = TransFusion(model_cfg, num_class, dataset)
        
#         self.ref_model.module_list = self.ref_model.build_networks()
#         self.model.module_list = self.model.build_networks()
#         self.register_buffer('global_step', torch.LongTensor(1).zero_())
        
#         for param in self.ref_model.parameters():
#             param.requires_grad = False
            
#         for name, param in self.model.named_parameters():
#             if 'cross_attention_layer' in name:
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False
        
        
#     def _load_state_dict(self, model_state_disk, *, strict=True):
#         state_dict = self.state_dict()  # local cache of state_dict

#         spconv_keys = find_all_spconv_keys(self)

#         update_model_state = {}
#         for key, val in model_state_disk.items():
#             if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
#                 # with different spconv versions, we need to adapt weight shapes for spconv blocks
#                 # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

#                 val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
#                 if val_native.shape == state_dict[key].shape:
#                     val = val_native.contiguous()
#                 else:
#                     assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
#                     val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
#                     if val_implicit.shape == state_dict[key].shape:
#                         val = val_implicit.contiguous()

#             if key in state_dict and state_dict[key].shape == val.shape:
#                 update_model_state[key] = val
#                 # logger.info('Update weight %s: %s' % (key, str(val.shape)))

#         if strict:
#             self.load_state_dict(update_model_state)
#         else:
#             state_dict.update(update_model_state)
#             self.load_state_dict(state_dict)
#         return state_dict, update_model_state
    
#     @property
#     def mode(self):
#         return 'TRAIN' if self.training else 'TEST'

#     def update_global_step(self):
#         self.global_step += 1
        
    
#     def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
#         if not os.path.isfile(filename):
#             raise FileNotFoundError

#         logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
#         loc_type = torch.device('cpu') if to_cpu else None
#         checkpoint = torch.load(filename, map_location=loc_type)
#         model_state_disk = checkpoint['model_state']
#         if not pre_trained_path is None:
#             pretrain_checkpoint = torch.load(pre_trained_path, map_location=loc_type)
#             pretrain_model_state_disk = pretrain_checkpoint['model_state']
#             model_state_disk.update(pretrain_model_state_disk)

#         version = checkpoint.get("version", None)
#         if version is not None:
#             logger.info('==> Checkpoint trained from version: %s' % version)

#         state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

#         for key in state_dict:
#             if key not in update_model_state:
#                 logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

#         logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

#     def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        
#         if not os.path.isfile(filename):
#             raise FileNotFoundError

#         logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
#         loc_type = torch.device('cpu') if to_cpu else None
#         checkpoint = torch.load(filename, map_location=loc_type)
#         epoch = checkpoint.get('epoch', -1)
#         it = checkpoint.get('it', 0.0)

#         self._load_state_dict(checkpoint['model_state'], strict=True)

#         if optimizer is not None:
#             if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
#                 logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
#                             % (filename, 'CPU' if to_cpu else 'GPU'))
#                 optimizer.load_state_dict(checkpoint['optimizer_state'])
#             else:
#                 assert filename[-4] == '.', filename
#                 src_file, ext = filename[:-4], filename[-3:]
#                 optimizer_filename = '%s_optim.%s' % (src_file, ext)
#                 if os.path.exists(optimizer_filename):
#                     optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
#                     optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

#         if 'version' in checkpoint:
#             print('==> Checkpoint trained from version: %s' % checkpoint['version'])
#         logger.info('==> Done')

#         return it, epoch
        
        
#     def get_training_loss(self,batch_dict):
#         disp_dict = {}

#         loss_trans, tb_dict = batch_dict['loss'],batch_dict['tb_dict']
#         tb_dict = {
#             'loss_trans': loss_trans.item(),
#             **tb_dict
#         }

#         loss = loss_trans
#         return loss, tb_dict, disp_dict

#     def post_processing(self, batch_dict):
#         post_process_cfg = self.model.model_cfg.POST_PROCESSING
#         batch_size = batch_dict['batch_size']
#         final_pred_dict = batch_dict['final_box_dicts']
#         recall_dict = {}
#         for index in range(batch_size):
#             pred_boxes = final_pred_dict[index]['pred_boxes']

#             recall_dict = self.model.generate_recall_record(
#                 box_preds=pred_boxes,
#                 recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
#                 thresh_list=post_process_cfg.RECALL_THRESH_LIST
#             )

#         return final_pred_dict, recall_dict
    
        
#     def forward(self, batch_dict):
                
#         with torch.no_grad():
#             self.ref_model.eval()
            
#             no_grad_dict = batch_dict.copy()
#             no_grad_dict['points'] = no_grad_dict['prev_points']
            
#             for cur_module in self.ref_model.module_list:
#                 no_grad_dict = cur_module(no_grad_dict)
        
#         voxel_size = no_grad_dict['voxel_size']
#         batch_size = batch_dict['batch_size']
#         # gt_box_centers = batch_dict['gt_boxes'][...,:3] # [x, y, z], l, w, h, yaw, vx, vy, label
#         # predicted boxes -> [center(x,y), height, dim, rot, vel]
#         predicted_boxes = [pred_dict['pred_boxes'] for pred_dict in no_grad_dict['final_box_dicts']]
        
#         max_gt = max([len(x) for x in predicted_boxes])
#         batch_pred_boxes3d = torch.zeros((batch_size, max_gt, predicted_boxes[0].shape[-1])).to(predicted_boxes[0])
        
#         for k in range(batch_size):
#             batch_pred_boxes3d[k, :predicted_boxes[k].__len__(), :] = predicted_boxes[k]
        
#         batch_pred_boxes3d[...,:2] += (batch_pred_boxes3d[...,-2:] * 0.05)
    
#         # gt_boxes = batch_dict['gt_boxes']
                                
#         voxel_center_points, _ = get_voxel_centers_in_boxes(batch_pred_boxes3d, voxel_size)
#         guidance_points = voxel_center_points.reshape(batch_size, -1, 3)      
#         non_zero_mask = guidance_points.abs().sum(dim=2) != 0
#         non_zero_indices = non_zero_mask.nonzero(as_tuple=False)
        
#         batch_dict['non_zero_box_indices'] =  non_zero_indices
#         batch_dict['box_centers'] = guidance_points
#         batch_dict['bev_features'] = no_grad_dict['spatial_features']
                
#         for cur_module in self.model.module_list:
#             batch_dict = cur_module(batch_dict)                
            
#         if self.training:
#             loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

#             ret_dict = {
#                 'loss': loss
#             }
#             return ret_dict, tb_dict, disp_dict
#         else:
#             pred_dicts, recall_dicts = self.post_processing(batch_dict)
#             return pred_dicts, recall_dicts


# class TransFusionwithClassLabel(Detector3DTemplate):
#     def __init__(self, model_cfg, num_class, dataset):
#         super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
#         self.module_list = self.build_networks()

#     def forward(self, batch_dict):
        

#         for cur_module in self.module_list:
#             batch_dict = cur_module(batch_dict)

#         if self.training:
#             loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

#             ret_dict = {
#                 'loss': loss
#             }
#             return ret_dict, tb_dict, disp_dict
#         else:
#             pred_dicts, recall_dicts = self.post_processing(batch_dict)
#             return pred_dicts, recall_dicts

#     def get_training_loss(self,batch_dict):
#         disp_dict = {}

#         loss_trans, tb_dict = batch_dict['loss'],batch_dict['tb_dict']
#         tb_dict = {
#             'loss_trans': loss_trans.item(),
#             **tb_dict
#         }

#         loss = loss_trans
#         return loss, tb_dict, disp_dict

#     def post_processing(self, batch_dict):
#         post_process_cfg = self.model_cfg.POST_PROCESSING
#         batch_size = batch_dict['batch_size']
#         final_pred_dict = batch_dict['final_box_dicts']
#         recall_dict = {}
#         for index in range(batch_size):
#             pred_boxes = final_pred_dict[index]['pred_boxes']

#             recall_dict = self.generate_recall_record(
#                 box_preds=pred_boxes,
#                 recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
#                 thresh_list=post_process_cfg.RECALL_THRESH_LIST
#             )

#         return final_pred_dict, recall_dict
