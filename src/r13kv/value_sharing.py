"""Validated CPU fit-shard loader; suitable for a frozen method descriptor."""
import hashlib,torch
def load_rotation_shards(root,spec,text_config):
    mode=spec['layout'];assert mode in ['private_layer_head_modality','shared_layer_modality']
    layers=text_config.num_hidden_layers;heads=text_config.num_key_value_heads
    dim=text_config.hidden_size//text_config.num_attention_heads
    expected=(layers,heads,dim,dim) if mode.startswith('private') else (layers,dim,dim)
    assert len(spec['shards'])==3;result={};seen=set()
    for part in spec['shards']:
        mod=part['modality'];assert mod in (0,1,2) and mod not in seen;seen.add(mod)
        path=root/part['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==part['SHA256']
        fit=torch.load(path,map_location='cpu',weights_only=True);value=fit['rotations']
        assert fit['modality']==mod and fit['mode']==('private' if mode.startswith('private') else 'shared')
        assert value.shape==expected and value.dtype==torch.float32 and torch.isfinite(value).all()
        assert float((value@value.transpose(-1,-2)-torch.eye(dim)).abs().max())<2e-5
        for layer in range(layers):
            for head in range(heads):result[layer,head,mod]=value[layer,head] if mode.startswith('private') else value[layer]
    return result
