import json
import time
import torch
import argparse
import gradio as gr

from GenMM import GenMM
from nearest_neighbor.losses import PatchCoherentLoss
from dataset.tracks_motion import TracksMotion

args = argparse.ArgumentParser(description='Web server for GenMM')
args.add_argument('-d', '--device', default="cuda:0", type=str, help='device to use.')
args.add_argument('--ip', default="0.0.0.0", type=str, help='interface url to host.')
args.add_argument('--port', default=8000, type=int, help='interface port to serve.')
args.add_argument('--debug', action='store_true', help='debug mode.')
args = args.parse_args()

def generate(data):
    data = json.loads(data)

    # create track object
    motion_data = [TracksMotion(data['tracks'], repr='repr6d', use_velo=True, keep_y_pos=True, padding_last=False)]
    model = GenMM(device=args.device, silent=True)
    criteria = PatchCoherentLoss(patch_size=data['setting']['patch_size'], 
                                alpha=data['setting']['alpha'] if data['setting']['completeness'] else None, 
                                loop=data['setting']['loop'], cache=True)

    # start generation
    start = time.time()
    syn = model.run(motion_data, criteria,
                    num_frames=str(data['setting']['frames']),
                    num_steps=data['setting']['num_steps'],
                    noise_sigma=data['setting']['noise_sigma'],
                    patch_size=data['setting']['patch_size'], 
                    coarse_ratio=f'{data["setting"]["patch_size"]}x_nframes',
                    # coarse_ratio=f'3x_patchsize',
                    pyr_factor=data['setting']['pyr_factor'])
    end = time.time()

    data['time'] = end - start
    data['tracks'] = motion_data[0].parse(syn)

    return data

if __name__ == '__main__':
    demo = gr.Interface(fn=generate, inputs="json", outputs="json")
    demo.launch(debug=args.debug, server_name=args.ip, server_port=args.port)