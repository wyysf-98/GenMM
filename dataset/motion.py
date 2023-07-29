import torch
import torch.nn.functional as F


class MotionData:
    def __init__(self, data, repr='quat', use_velo=True, keep_y_pos=True, padding_last=False, contact_id=None):
        '''
        BaseMotionData constructor
        Args:
            data         : torch.Tensor, [batch_size x n_channels x n_frames] input motion data, 
                           the channels dim shoud be [n_joints x n_dim_of_rotation + 3(global position)]
            repr         : string, rotation representation, support ['quat', 'repr6d', 'euler'] 
            use_velo     : book, whether to transform the joints positions to velocities
            keep_y_pos   : bool, whether to keep y position when converting to velocity
            padding_last : bool, whether to pad the last position
            contact_id   : list, contact joints id
        '''
        self.data = data 
        self.repr = repr
        self.use_velo = use_velo
        self.keep_y_pos = keep_y_pos
        self.padding_last = padding_last
        self.contact_id = contact_id
        self.begin_pos = None

        # assert the rotation representation
        if self.repr == 'quat':
            self.n_rot = 4
            assert (self.data.shape[1] - 3) % 4 == 0, 'rotation is not "quaternion" representation'
        elif self.repr == 'repr6d':
            self.n_rot = 6
            assert (self.data.shape[1] - 3) % 6 == 0, 'rotation is not "repr6d" representation'
        elif self.repr == 'eluer':
            self.n_rot = 3
            assert (self.data.shape[1] - 3) % 3 == 0, 'rotation is not "euler" representation'

        # whether to pad the position data with zero
        if self.padding_last:
            self.n_pad = self.data.shape[1] - 3  # pad position channels to match the n_channels of rotation
            paddings = torch.zeros_like(self.data[:, :self.n_pad])
            self.data = torch.cat((self.data, paddings), dim=1)
        else:
            self.n_pad = 0

        # get the contact information
        if self.contact_id is not None:
            self.n_contact = len(contact_id)
        else:
            self.n_contact = 0

        # whether to keep y position when converting to velocity
        if self.keep_y_pos:
            self.velo_mask = [-3, -1]
        else:
            self.velo_mask = [-3, -2, -1]

        # whether to convert global position to velocity
        if self.use_velo:
            self.data =  self.to_velocity(self.data)


    def __len__(self):
        '''
        return the number of motion frames
        '''
        return self.data.shape[-1]


    def sample(self, size=None, slerp=False):
        '''
        sample the motion data using given size
        '''
        if size is None:
            return self.data
        else:
            if slerp:
                motion = self.slerp(self.data, size=size)
            else:
                motion = F.interpolate(self.data, size=size, mode='linear', align_corners=False)
            return motion


    def to_velocity(self, pos):
        '''
        convert motion data to velocity
        '''
        assert self.begin_pos is None, 'the motion data had been converted to velocity'
        msk = [i - self.n_pad for i in self.velo_mask]
        velo = pos.detach().clone().to(pos.device)
        velo[:, msk, 1:] = pos[:, msk, 1:] - pos[:, msk, :-1]
        self.begin_pos = pos[:, msk, 0].clone()
        velo[:, msk, 0] = pos[:, msk, 1]
        return velo

    def to_position(self, velo):
        '''
        convert motion data to position
        '''
        assert self.begin_pos is not None, 'the motion data is already position'
        msk = [i - self.n_pad for i in self.velo_mask]
        pos = velo.detach().clone().to(velo.device)
        pos[:, msk, 0] = self.begin_pos.to(velo.device)
        pos[:, msk] = torch.cumsum(pos[:, msk], dim=-1)
        self.begin_pos = None
        return pos