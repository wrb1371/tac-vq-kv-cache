"""Load a pinned, paid, fixed-stage CPU K-transform artifact."""
import hashlib,torch
def load_key_rotation(root,spec,text_config):
    path=root/spec['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['SHA256']
    data=torch.load(path,map_location='cpu',weights_only=True);value=data['rotation']
    d=text_config.hidden_size//text_config.num_attention_heads;h=text_config.num_key_value_heads
    assert data['seed']==131042 and data['stages']==spec['stages'] and data['stages'] in (1,2)
    assert value.shape==(h,d,d) and value.dtype==torch.float32 and torch.isfinite(value).all()
    assert float((value@value.transpose(-1,-2)-torch.eye(d)).abs().max())<2e-6
    return value
