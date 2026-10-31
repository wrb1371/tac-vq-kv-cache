"""Same frozen7B runtime with an explicit26GiB execution cap; GPU1 excluded."""
import os
import torch
from ar26kv.runtime import load as original_load

def load():
    device=os.environ.get('CUDA_VISIBLE_DEVICES')
    assert device and ',' not in device and int(device) in [0,2,3,4,6,7],device
    cap=26*2**30/torch.cuda.get_device_properties(0).total_memory
    original=torch.cuda.set_per_process_memory_fraction
    def bounded(fraction,device=None):return original(min(fraction,cap),device)
    torch.cuda.set_per_process_memory_fraction=bounded
    try:return original_load()
    finally:
        torch.cuda.set_per_process_memory_fraction=original
        original(cap)
