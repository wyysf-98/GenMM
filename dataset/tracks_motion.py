import os
from os.path import join as pjoin
import numpy as np
import copy
import torch
import torch.nn.functional as F
from utils.transforms import quat2repr6d, quat2euler, repr6d2quat

class TracksParser():
    def __init__(self, tracks_json, scale=1.0, requires_contact=False, joint_reduction=False):
        assert requires_contact==False, 'contact is not implemented for tracks data yet!!!'

        self.tracks_json = tracks_json
        self.scale = scale
        self.requires_contact = requires_contact
        self.joint_reduction = joint_reduction
        
        self.skeleton_names = []
        self.rotations = []
        for i, track in enumerate(self.tracks_json):
            # print(i, track['name'])
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

        if self.requires_contact:
            virtual_contact = torch.zeros_like(rotations[:, :len(self.skeleton.contact_id)])
            virtual_contact[..., 0] = self.contact_label
            rotations = torch.cat([rotations, virtual_contact], dim=1)

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
    def __init__(self, tracks_json, scale=1.0, repr='repr6d', padding=False,
                 use_velo=True, contact=False, keep_y_pos=True, joint_reduction=False):
        self.scale = scale
        self.tracks = TracksParser(tracks_json, scale, requires_contact=contact, joint_reduction=joint_reduction)
        self.raw_motion = self.tracks.to_tensor(repr=repr)
        self.extra = {

        }

        self.repr = repr
        if repr == 'quat':
            self.n_rot = 4
        elif repr == 'repr6d':
            self.n_rot = 6
        elif repr == 'euler':
            self.n_rot = 3
        self.padding = padding
        self.use_velo = use_velo
        self.contact = contact
        self.keep_y_pos = keep_y_pos
        self.joint_reduction = joint_reduction

        self.raw_motion = self.raw_motion.permute(1, 0).unsqueeze_(0) # Shape = (1, n_channel, n_frames)
        self.extra['global_pos'] = self.raw_motion[:, -3:, :]

        if padding:
            self.n_pad = self.n_rot - 3 # pad position channels
            paddings = torch.zeros_like(self.raw_motion[:, :self.n_pad])
            self.raw_motion = torch.cat((self.raw_motion, paddings), dim=1)
        else:
            self.n_pad = 0
        self.raw_motion = torch.cat((self.raw_motion[:, :-3-self.n_pad], self.raw_motion[:, -3-self.n_pad:]), dim=1)

        if self.use_velo:
            self.msk = [-3, -2, -1] if not keep_y_pos else [-3, -1]
            self.raw_motion = self.pos2velo(self.raw_motion)

        self.n_contact = len(self.tracks.skeleton.contact_id) if contact else 0

    @property
    def n_channels(self):
        return self.raw_motion.shape[1]

    def __len__(self):
        return self.raw_motion.shape[-1]

    def pos2velo(self, pos):
        msk = [i - self.n_pad for i in self.msk]
        velo = pos.detach().clone().to(pos.device)
        velo[:, msk, 1:] = pos[:, msk, 1:] - pos[:, msk, :-1]
        self.begin_pos = pos[:, msk, 0].clone()
        velo[:, msk, 0] = pos[:, msk, 1]
        return velo

    def velo2pos(self, velo):
        msk = [i - self.n_pad for i in self.msk]
        pos = velo.detach().clone().to(velo.device)
        pos[:, msk, 0] = self.begin_pos.to(velo.device)
        pos[:, msk] = torch.cumsum(velo[:, msk], dim=-1)
        return pos

    def motion2pos(self, motion):
        if not self.use_velo:
            return motion
        else:
            self.velo2pos(motion.clone())

    def sample(self, size=None, slerp=False, align_corners=False):
        if size is None:
            return {'motion': self.raw_motion, 'extra': self.extra}
        else:
            if slerp:
                raise NotImplementedError('slerp is not not implemented yet!!!')
            else:
                motion = F.interpolate(self.raw_motion, size=size, mode='linear', align_corners=align_corners)
                extra = {}
                if 'global_pos' in self.extra.keys():
                    extra['global_pos'] = F.interpolate(self.extra['global_pos'], size=size, mode='linear', align_corners=align_corners)

            return motion
            # return {'motion': motion, 'extra': extra}

    def parse(self, motion, keep_velo=False,):
        """
        No batch support here!!!
        :returns tracks_json
        """
        motion = motion.clone()

        if self.use_velo and not keep_velo:
            motion = self.velo2pos(motion)
        if self.n_pad:
            motion = motion[:, :-self.n_pad]
        if self.contact:
            raise NotImplementedError('contact is not implemented yet!!!')

        motion = motion.squeeze().permute(1, 0)
        pos = motion[..., -3:] / self.scale
        rot = motion[..., :-3].reshape(motion.shape[0], -1, self.n_rot)
        if self.repr == 'repr6d':
            rot = repr6d2quat(rot)
        elif self.repr == 'euler':
            raise NotImplementedError('parse "euler is not implemented yet!!!')

        times = []
        out_tracks_json = copy.deepcopy(self.tracks.tracks_json)
        for i, _track in enumerate(out_tracks_json):
            if i == 0:
                times = [ j * out_tracks_json[i]['times'][1] for j in range(motion.shape[0])]
                out_tracks_json[i]['values'] = pos.flatten().detach().cpu().numpy().tolist() 
            else:
                out_tracks_json[i]['values'] = rot[:, i-1, :].flatten().detach().cpu().numpy().tolist()
            out_tracks_json[i]['times'] = times

        return out_tracks_json
