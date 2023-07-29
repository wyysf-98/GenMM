import os
from os.path import join as pjoin
import numpy as np
import copy
import torch
from .motion import MotionData
from ..utils.transforms import quat2repr6d, quat2euler, repr6d2quat

class TracksParser():
    def __init__(self, tracks_json, scale):
        self.tracks_json = tracks_json
        self.scale = scale
        
        self.skeleton_names = []
        self.rotations = []
        for i, track in enumerate(self.tracks_json):
            self.skeleton_names.append(track['name'])
            if i == 0:
                assert track['type'] == 'vector'
                self.position = np.array(track['values']).reshape(-1, 3) * self.scale
                self.num_frames = self.position.shape[0]
            else:
                assert track['type'] == 'quaternion' # DEAFULT: quaternion
                rotation = np.array(track['values']).reshape(-1, 4)
                if rotation.shape[0] == 0:
                    rotation = np.zeros((self.num_frames, 4))
                elif rotation.shape[0] < self.num_frames:
                    rotation = np.repeat(rotation, self.num_frames // rotation.shape[0], axis=0)
                elif rotation.shape[0] > self.num_frames:
                    rotation = rotation[:self.num_frames]
                self.rotations += [rotation]
        self.rotations = np.array(self.rotations, dtype=np.float32)

    def to_tensor(self, repr='euler', rot_only=False):
        if repr not in ['euler', 'quat', 'quaternion', 'repr6d']:
            raise Exception('Unknown rotation representation')
        rotations = self.get_rotation(repr=repr)
        positions = self.get_position()

        if rot_only:
            return rotations.reshape(rotations.shape[0], -1)

        rotations = rotations.reshape(rotations.shape[0], -1)
        return torch.cat((rotations, positions), dim=-1)

    def get_rotation(self, repr='quat'):
        if repr == 'quaternion' or repr == 'quat' or repr == 'repr6d':
            rotations = torch.tensor(self.rotations, dtype=torch.float).transpose(0, 1)
        if repr == 'repr6d':
            rotations = quat2repr6d(rotations)
        if repr == 'euler':
            rotations = quat2euler(rotations)
        return rotations

    def get_position(self):
        return torch.tensor(self.position, dtype=torch.float32)

class TracksMotion:
    def __init__(self, tracks_json, scale=1.0, repr='quat', use_velo=True, keep_up_pos=True, up_axis='Y_UP', padding_last=False):
        '''
        TracksMotion constructor
        Args:
            tracks_json      : dict, json format tracks data to load from
            scale            : float, scale of the tracks motion data
            repr             : string, rotation representation, support ['quat', 'repr6d', 'euler'] 
            use_velo         : book, whether to transform the joints positions to velocities
            keep_up_pos      : bool, whether to keep y position when converting to velocity
            up_axis          : string, string, up axis of the motion data
            padding_last     : bool, whether to pad the last position
        '''
        self.tracks_json = tracks_json

        self.raw_data = TracksParser(tracks_json, scale)
        self.motion_data = MotionData(self.raw_data.to_tensor(repr=repr).permute(1, 0).unsqueeze(0), repr=repr, use_velo=use_velo, keep_up_pos=keep_up_pos, up_axis=up_axis, 
                                      padding_last=padding_last, contact_id=None)
    @property
    def repr(self):
        return self.motion_data.repr

    @property
    def use_velo(self):
        return self.motion_data.use_velo

    @property
    def keep_up_pos(self):
        return self.motion_data.keep_up_pos
    
    @property
    def padding_last(self):
        return self.motion_data.padding_last

    @property
    def n_pad(self):
        return self.motion_data.n_pad

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
        pos = motion[..., -3:] / self.raw_data.scale
        rot = motion[..., :-3].reshape(motion.shape[0], -1, self.n_rot)
        if self.repr == 'repr6d':
            rot = repr6d2quat(rot)
        elif self.repr == 'euler':
            raise NotImplementedError('parse "euler is not implemented yet!!!')

        times = []
        out_tracks_json = copy.deepcopy(self.tracks_json)
        for i, _track in enumerate(out_tracks_json):
            if i == 0:
                times = [ j * out_tracks_json[i]['times'][1] for j in range(motion.shape[0])]
                out_tracks_json[i]['values'] = pos.flatten().detach().cpu().numpy().tolist() 
            else:
                out_tracks_json[i]['values'] = rot[:, i-1, :].flatten().detach().cpu().numpy().tolist()
            out_tracks_json[i]['times'] = times

        return out_tracks_json
