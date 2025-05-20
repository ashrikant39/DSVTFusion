from torch.utils.checkpoint import checkpoint
from .detector3d_template import Detector3DTemplate
import torch
import torch.nn as nn
import os
from ...utils.spconv_utils import find_all_spconv_keys

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
