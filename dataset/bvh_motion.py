import os
import os.path as osp
import torch
import numpy as np
import torch.nn.functional as F
from .motion import MotionData
from .bvh.bvh_parser import BVH_file


## Some skeleton configurations
crab_dance_corps_names = ['ORG_Hips', 'ORG_BN_Bip01_Pelvis', 'DEF_BN_Eye_L_01', 'DEF_BN_Eye_L_02', 'DEF_BN_Eye_L_03', 'DEF_BN_Eye_L_03_end', 'DEF_BN_Eye_R_01', 'DEF_BN_Eye_R_02', 'DEF_BN_Eye_R_03', 'DEF_BN_Eye_R_03_end', 'DEF_BN_Leg_L_11', 'DEF_BN_Leg_L_12', 'DEF_BN_Leg_L_13', 'DEF_BN_Leg_L_14', 'DEF_BN_Leg_L_15', 'DEF_BN_Leg_L_15_end', 'DEF_BN_Leg_R_11', 'DEF_BN_Leg_R_12', 'DEF_BN_Leg_R_13', 'DEF_BN_Leg_R_14', 'DEF_BN_Leg_R_15', 'DEF_BN_Leg_R_15_end', 'DEF_BN_leg_L_01', 'DEF_BN_leg_L_02', 'DEF_BN_leg_L_03', 'DEF_BN_leg_L_04', 'DEF_BN_leg_L_05', 'DEF_BN_leg_L_05_end',
                         'DEF_BN_leg_L_06', 'DEF_BN_Leg_L_07', 'DEF_BN_Leg_L_08', 'DEF_BN_Leg_L_09', 'DEF_BN_Leg_L_10', 'DEF_BN_Leg_L_10_end', 'DEF_BN_leg_R_01', 'DEF_BN_leg_R_02', 'DEF_BN_leg_R_03', 'DEF_BN_leg_R_04', 'DEF_BN_leg_R_05', 'DEF_BN_leg_R_05_end', 'DEF_BN_leg_R_06', 'DEF_BN_Leg_R_07', 'DEF_BN_Leg_R_08', 'DEF_BN_Leg_R_09', 'DEF_BN_Leg_R_10', 'DEF_BN_Leg_R_10_end', 'DEF_BN_Bip01_Pelvis', 'DEF_BN_Bip01_Pelvis_end', 'DEF_BN_Arm_L_01', 'DEF_BN_Arm_L_02', 'DEF_BN_Arm_L_03', 'DEF_BN_Arm_L_03_end', 'DEF_BN_Arm_R_01', 'DEF_BN_Arm_R_02', 'DEF_BN_Arm_R_03', 'DEF_BN_Arm_R_03_end']
skeleton_confs = {
    'mixamo': {
        'corps_names': ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToe_End', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToe_End', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand'],
        'contact_names': ['LeftToe_End', 'RightToe_End', 'LeftToeBase', 'RightToeBase'],
        'contact_threshold': 0.018
    },
    'crab_dance': {
        'corps_names': crab_dance_corps_names,
        'contact_names': [name for name in crab_dance_corps_names if 'end' in name and ('05' in name or '10' in name or '15' in name)],
        'contact_threshold': 0.006
    },
    'xia': {
        'corps_names': ['Hips', 'LHipJoint', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RHipJoint', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LowerBack', 'Spine', 'Spine1', 'Neck', 'Neck1', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftFingerBase', 'LeftHandIndex1', 'LThumb', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightFingerBase', 'RightHandIndex1', 'RThumb'],
        'contact_names': ['LeftToeBase', 'RightToeBase'],
        'contact_threshold': 0.006
    }
}

class BVHMotion:
    def __init__(self, bvh_file, skeleton_name=None, repr='quat', use_velo=True, keep_y_pos=False, padding_last=False, requires_contact=False, joint_reduction=False):
        '''
        BVHMotion constructor
        Args:
            bvh_file         : string, bvh_file path to load from
            skelton_name     : string, name of predefined skeleton, used when joint_reduction==True or contact==True
            repr             : string, rotation representation, support ['quat', 'repr6d', 'euler'] 
            use_velo         : book, whether to transform the joints positions to velocities
            keep_y_pos       : bool, whether to keep y position when converting to velocity
            padding_last     : bool, whether to pad the last position
            skeleton_conf    : dict, provide configuration for contact detection and joint detection
            requires_contact : bool, whether to concatenate contact information
            joint_reduction  : bool, whether to reduce the joint number
        '''
        self.bvh_file = bvh_file
        self.skeleton_name = skeleton_name
        if skeleton_name is not None:
            assert skeleton_name in skeleton_confs, f'{skeleton_name} not found, please add a skeleton configuration.'
        self.requires_contact = requires_contact
        self.joint_reduction = joint_reduction

        self.raw_data = BVH_file(bvh_file, skeleton_confs[skeleton_name] if skeleton_name is not None else None, requires_contact, joint_reduction, auto_scale=True)
        self.motion_data = MotionData(self.raw_data.to_tensor(repr=repr).permute(1, 0).unsqueeze(0), repr=repr, use_velo=use_velo, keep_y_pos=keep_y_pos,
                                      padding_last=padding_last, contact_id=self.raw_data.skeleton.contact_id if requires_contact else None)
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
    def n_concat(self):
        return self.motion_data.n_contact

    @property
    def n_rot(self):
        return self.motion_data.n_rot

    def sample(self, size=None, slerp=False):
        '''
        Sample motion data, support slerp
        '''
        return self.motion_data.sample(size, slerp)


    def write(self, filename, data):
        '''
        Parse motion data into position, velocity and contact(if exists)
        data should be []
        No batch support here!!!
        '''
        assert len(data.shape) == 3, 'The data format should be [batch_size x n_channels x n_frames]' 

        if self.n_pad:
            data = data.clone()[:, :-self.n_pad]
        if self.use_velo:
            data = self.motion_data.to_position(data)
        data = data.squeeze().permute(1, 0)
        pos = data[..., -3:]
        rot = data[..., :-3].reshape(data.shape[0], -1, self.n_rot)
        if self.requires_contact:
            contact = rot[..., -self.n_contact:, 0]
            rot = rot[..., :-self.n_contact, :]
        else:
            contact = None

        if contact is not None:
            np.save(filename + '.contact', contact.detach().cpu().numpy())

        self.raw_data.writer.write(filename, rot, pos, names=self.raw_data.skeleton.names, repr=self.repr)


def load_multiple_dataset(name_list, **kargs):
        with open(name_list, 'r') as f:
            names = [line.strip() for line in f.readlines()]
        datasets = []
        for f in names:
            kargs['bvh_file'] = osp.join(name_list, f)
            datasets.append(MotionData(**kargs))
        return datasets