import os
import os.path as osp
import torch
import numpy as np
import torch.nn.functional as F
from .motion import MotionData
from utils.transforms import quat2repr6d, euler2mat, mat2quat, repr6d2quat, quat2euler

class BlenderMotion:
    def __init__(self, motion_data, repr='quat', use_velo=True, keep_y_pos=False, padding_last=False):
        '''
        BVHMotion constructor
        Args:
            motion_data      : np.array, bvh format data to load from
            repr             : string, rotation representation, support ['quat', 'repr6d', 'euler'] 
            use_velo         : book, whether to transform the joints positions to velocities
            keep_y_pos       : bool, whether to keep y position when converting to velocity
            padding_last     : bool, whether to pad the last position
            requires_contact : bool, whether to concatenate contact information
        '''
        self.motion_data = motion_data


        def to_tensor(motion_data, repr='euler', auto_scale=False, rot_only=False):
            if repr not in ['euler', 'quat', 'quaternion', 'repr6d']:
                raise Exception('Unknown rotation representation')
            if repr == 'quaternion' or repr == 'quat' or repr == 'repr6d': # default is euler for blender data
                rotations = torch.tensor(motion_data[:, 3:], dtype=torch.float).view(motion_data.shape[0], -1, 3)
            if repr == 'quat':
                rotations = euler2mat(rotations)
                rotations = mat2quat(rotations)
            if repr == 'repr6d':
                rotations = euler2mat(rotations)
                rotations = mat2quat(rotations)
                rotations = quat2repr6d(rotations)

            positions = torch.tensor(motion_data[:, :3], dtype=torch.float32)

            if rot_only:
                return rotations.reshape(rotations.shape[0], -1)

            rotations = rotations.reshape(rotations.shape[0], -1)
            return torch.cat((rotations, positions), dim=-1)
        
        self.motion_data = MotionData(to_tensor(motion_data, repr=repr).permute(1, 0).unsqueeze(0), repr=repr, use_velo=use_velo, keep_y_pos=keep_y_pos,
                                      padding_last=padding_last, contact_id=None)
    @property
    def repr(self):
        return self.motion_data.repr

    @property
    def use_velo(self):
        return self.motion_data.use_velo

    @property
    def keep_y_pos(self):
        return self.motion_data.keep_y_pos
    
    @property
    def padding_last(self):
        return self.motion_data.padding_last
    
    @property
    def concat_id(self):
        return self.motion_data.contact_id
    
    @property
    def n_pad(self):
        return self.motion_data.n_pad
    
    @property
    def n_contact(self):
        return self.motion_data.n_contact

    @property
    def n_rot(self):
        return self.motion_data.n_rot

    def sample(self, size=None, slerp=False):
        '''
        Sample motion data, support slerp
        '''
        return self.motion_data.sample(size, slerp)

    def parse(self, motion, keep_velo=False,):
        """
        No batch support here!!!
        :returns tracks_json
        """
        motion = motion.clone()

        if self.use_velo and not keep_velo:
            motion = self.motion_data.to_position(motion)
        if self.n_pad:
            motion = motion[:, :-self.n_pad]

        motion = motion.squeeze().permute(1, 0)
        pos = motion[..., -3:]
        rot = motion[..., :-3].reshape(motion.shape[0], -1, self.n_rot)
        if self.repr == 'quat':
            rot = quat2euler(rot)
        elif self.repr == 'repr6d':
            rot = repr6d2quat(rot)
            rot = quat2euler(rot)

        return torch.cat([pos, rot.view(motion.shape[0], -1)], dim=-1).cpu().numpy()
