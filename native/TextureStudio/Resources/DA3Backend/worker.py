"""Texture Studio isolated, offline DA3-GIANT-1.1 inference worker.

Input is a bounded RGB PNG. Output is untouched top-down little-endian Float32
relative camera-Z depth; surface processing and display normalization live in
Swift. No old Studio source, API, network, CUDA or CPU fallback is used.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import resource

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['DA3_LOG_LEVEL'] = 'ERROR'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'upstream'))
REVISION = '72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19'
UPSTREAM = '3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'
WEIGHTS_HASH = '1e47a08338ca73a6d6a21d37fd060b26b993b672bc6ddf6295fe474df2592001'
CONFIG_HASH = '74626a50d6dee2a11820291a4305c1a34aa5adc4f7260908bbdbc9a939ba8e93'


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def emit(**data):
    print(json.dumps(data), flush=True)


def probe():
    import torch
    import torchvision
    import cv2
    import numpy
    import safetensors
    from depth_anything_3.cfg import create_object
    from depth_anything_3.utils.io.input_processor import InputProcessor
    from depth_anything_3.model.da3 import DepthAnything3Net
    from depth_anything_3.model.gs_adapter import GaussianAdapter
    if not torch.backends.mps.is_available():
        raise RuntimeError('This Python environment has no usable PyTorch MPS backend. Use an Apple Silicon Mac and a Metal-enabled PyTorch build.')
    return {'python': sys.version.split()[0], 'torch': torch.__version__, 'device': 'mps', 'upstreamRevision': UPSTREAM}


def depth_only(network, tensor):
    """Evaluate unchanged primary DualDPT branch, omitting independent ray heads.

    The projected pyramid and primary refinement/activation match upstream
    DualDPT. Camera, ray and Gaussian outputs do not feed the depth branch.
    """
    from depth_anything_3.model.utils.head_utils import custom_interpolate
    head = network.head
    if head.__class__.__name__ != 'DualDPT':
        raise RuntimeError('The pinned GIANT 1.1 primary decoder is not DualDPT.')
    height, width = tensor.shape[-2:]
    feats, _ = network.backbone(tensor, cam_token=None, export_feat_layers=[], ref_view_strategy='first')
    batch, views, count, channels = feats[0][0].shape
    patch_h, patch_w = height // head.patch_size, width // head.patch_size
    resized = []
    for stage, take in enumerate(head.intermediate_layer_idx):
        value = head.norm(feats[take][0].reshape(batch * views, count, channels))
        value = value.permute(0, 2, 1).reshape(batch * views, channels, patch_h, patch_w)
        value = head.projects[stage](value)
        if head.pos_embed:
            value = head._add_pos_embed(value, width, height)
        resized.append(head.resize_layers[stage](value))
    del feats
    scratch = head.scratch
    l1, l2, l3, l4 = (getattr(scratch, f'layer{stage}_rn')(value) for stage, value in enumerate(resized, 1))
    del resized
    value = scratch.refinenet4(l4, size=l3.shape[2:]); del l4
    value = scratch.refinenet3(value, l3, size=l2.shape[2:]); del l3
    value = scratch.refinenet2(value, l2, size=l1.shape[2:]); del l2
    value = scratch.refinenet1(value, l1); del l1
    value = scratch.output_conv1(value)
    value = custom_interpolate(value, (int(height / head.down_ratio), int(width / head.down_ratio)), mode='bilinear', align_corners=True)
    if head.pos_embed:
        value = head._add_pos_embed(value, width, height)
    logits = scratch.output_conv2(value).permute(0, 2, 3, 1)
    depth = head._apply_activation_single(logits[..., :-1], head.activation).squeeze(-1)
    return depth.reshape(batch, views, *depth.shape[1:])


def predict(args):
    started = time.monotonic()
    import numpy as np
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    from safetensors.torch import load_model
    from depth_anything_3.cfg import create_object
    from depth_anything_3.utils.io.input_processor import InputProcessor
    if args.resolution not in (1036, 1540, 2044):
        raise ValueError('Choose an explicit supported inference edge: 1036, 1540 or 2044 pixels.')
    if not torch.backends.mps.is_available():
        raise RuntimeError('PyTorch MPS is unavailable; this app does not silently run GIANT on CPU.')
    folder = Path(args.model).resolve()
    config_path = folder / 'config.json'; weights = folder / 'model.safetensors'
    emit(progress=0.03, message='Verifying the exact GIANT 1.1 checkpoint')
    if config_path.stat().st_size != 1880 or sha256(config_path) != CONFIG_HASH:
        raise ValueError('config.json is not the pinned DA3-GIANT-1.1 configuration.')
    if weights.stat().st_size != 5422814644 or sha256(weights) != WEIGHTS_HASH:
        raise ValueError('model.safetensors is not the pinned DA3-GIANT-1.1 checkpoint. Locate or download the exact model.')
    config = json.loads(config_path.read_text())
    # Config identity is verified before OmegaConf dynamic object creation.
    class LocalDA3(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = create_object(OmegaConf.create(config['config']))
    emit(progress=0.15, message='Loading GIANT 1.1 on Apple Metal')
    network = LocalDA3()
    # Safetensors handles shared LayerNorm aliases, with strict key validation.
    missing, unexpected = load_model(network, str(weights), strict=True, device='cpu')
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint keys differ: missing={missing}, unexpected={unexpected}')
    network.eval().to(device='mps', dtype=torch.float32)
    image = Image.open(args.image).convert('RGB')
    images, _, _ = InputProcessor()([image], process_res=args.resolution,
        process_res_method='upper_bound_resize', sequential=True)
    tensor = images[None].to(device='mps', dtype=torch.float32)
    emit(progress=0.3, message='Predicting relative surface depth')
    with torch.inference_mode():
        depth = depth_only(network.model, tensor)
    torch.mps.synchronize()
    values = depth[0, 0].detach().to(device='cpu', dtype=torch.float32).numpy()
    if values.ndim != 2 or not np.isfinite(values).all():
        raise RuntimeError('The depth model returned an invalid or nonfinite map.')
    destination = Path(args.output); destination.mkdir(parents=True, exist_ok=True)
    values.astype('<f4', copy=False).tofile(destination / 'depth.f32')
    provenance = {'modelID': 'depth-anything/DA3-GIANT-1.1', 'revision': REVISION,
        'checkpointSHA256': WEIGHTS_HASH, 'upstreamRevision': UPSTREAM,
        'backend': 'PyTorch', 'device': 'mps', 'precision': 'Float32',
        'inputWidth': int(tensor.shape[-1]), 'inputHeight': int(tensor.shape[-2]),
        'processResolution': args.resolution, 'fullSourceFieldOfView': True,
        'rowOrder': 'top-down', 'elapsedSeconds': time.monotonic() - started,
        'residentPeakBytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        'metalAllocatedBytes': int(torch.mps.current_allocated_memory()),
        'metalDriverAllocatedBytes': int(torch.mps.driver_allocated_memory())}
    metadata = {'width': int(values.shape[1]), 'height': int(values.shape[0]),
        'outputName': 'depth', 'interpretation': 'relative_camera_z_depth: larger values are farther; unitless, no normalization',
        'provenance': provenance, 'rawMinimum': float(values.min()), 'rawMaximum': float(values.max()),
        'rawSHA256': sha256(destination / 'depth.f32')}
    (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    del tensor, images, network, depth
    torch.mps.empty_cache()
    emit(progress=1, message='Depth ready', result=metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--model'); parser.add_argument('--image'); parser.add_argument('--output')
    parser.add_argument('--resolution', type=int, default=1036)
    args = parser.parse_args()
    # Reserve stdout for JSON progress even if upstream imports print diagnostics.
    if args.probe:
        with contextlib.redirect_stdout(sys.stderr):
            info = probe()
        emit(ready=True, **info)
    else:
        if not all((args.model, args.image, args.output)):
            parser.error('--model, --image and --output are required')
        predict(args)

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        emit(error=str(error))
        raise SystemExit(1)
