"""Per-head occupancy-conditioned errors with the frozen decoded-anchor bank."""
import math

import torch

from r14kv.fitting import nearest
from ar26kv.matching import fixed_reference

FRACTIONS = [0., 1/32, 1/16, 1/8, 1/4, 1/2, 1.]
GOPS = [2, 4, 8, 16]


@torch.no_grad()
def prefixes(x, table, side, layer, independent=True, levels=6):
    mean = table[side+'_mean'][layer].float()[:, None] if independent else 0.
    scale = table[side+'_scale'][layer].float()[:, None]
    centered = x.float()-mean
    u = centered/scale
    gain = u.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6).bfloat16()
    residual = ((u/gain.float())@table['H']).reshape(1, -1, 16)
    total = torch.zeros_like(residual)
    answers = []
    for stage in range(levels):
        book = table[side+'_books'][stage][None].float()
        chosen = nearest(residual, book)
        codeword = torch.gather(book, 1, chosen[..., None].expand(-1, -1, 16))
        residual = residual-codeword
        total = total+codeword
        decoded = (total.reshape(x.shape)@table['H'].T)*gain.float()
        decoded = decoded*scale
        answers.append((decoded+mean).bfloat16())
    return answers


@torch.no_grad()
def measure_layer(layer, native, query_moments, value_metric, table, video, text,
                  grid, importance, sample_count=96):
    raw = {'K': native['K_pre'][0].cuda().float(), 'V': native['V'][0].cuda().float()}
    metrics = {'K': query_moments.cuda()/128., 'V': value_metric.cuda()}
    nt, h, w = grid
    p = h*w
    m = len(video)
    assert m == nt*p
    ordinal = torch.arange(m, device=video.device)
    qs = text[torch.linspace(0, len(text)-1, min(sample_count, len(text)), device=text.device).long()]
    query_errors = {}
    for side in ['K', 'V']:
        values = raw[side][:, qs]
        reconstructed = prefixes(values, table, side, layer)
        query_errors[side] = torch.stack([((values-y.float()).square()*metrics[side][:, None]).sum(-1).mean(-1)
                                          for y in reconstructed]).T
    output = {'norm': {}, 'sensitivity': {}}
    for gop in GOPS:
        anchor_ord = ordinal[(ordinal//p) % gop == 0]
        target_ord = ordinal[(ordinal//p) % gop != 0]
        anchor_sample = anchor_ord[torch.linspace(0, len(anchor_ord)-1,
                    min(sample_count, len(anchor_ord)), device=video.device).long()]
        target_sample = target_ord[torch.linspace(0, len(target_ord)-1,
                    min(sample_count, len(target_ord)), device=video.device).long()]
        aseq, tseq = video[anchor_sample].long(), video[target_sample].long()
        anchor_errors = {}
        for side in ['K', 'V']:
            values = raw[side][:, aseq]
            reconstructed = prefixes(values, table, side, layer)
            anchor_errors[side] = torch.stack([((values-y.float()).square()*metrics[side][:, None]).sum(-1).mean(-1)
                                                for y in reconstructed]).T
        for ranking in ['norm']:
            weights = None if ranking == 'norm' else [importance[s][layer].cuda() for s in ['K', 'V']]
            codes, scores = fixed_reference(raw, video, target_sample,
                    [table['K_scale'][layer], table['V_scale'][layer]], h, w, gop, weights)
            frame, yy, xx = target_sample//p, target_sample//w % h, target_sample % w
            ry = (yy[None]+codes.long()//3-1).clamp(0, h-1)
            rx = (xx[None]+codes.long() % 3-1).clamp(0, w-1)
            ref_ord = frame[None]//gop*gop*p+ry*w+rx
            references = video[ref_ord].long()
            order = torch.argsort(scores, dim=1, descending=True, stable=True)
            n = len(target_sample)
            record = {}
            for side in ['K', 'V']:
                anchors = raw[side].gather(1, references[..., None].expand(-1, -1, 128))
                target = raw[side][:, tseq]
                decoded_anchors = prefixes(anchors, table, side, layer)
                losses = torch.empty((len(FRACTIONS), 6, raw[side].shape[0]), device=video.device)
                for depth, prediction_bf16 in enumerate(decoded_anchors, start=1):
                    prediction = prediction_bf16.float()
                    innovation = target-prediction
                    corrected = (prediction+prefixes(innovation, table, side, layer,
                                                     independent=False, levels=depth)[-1].float()).bfloat16().float()
                    copy_error = (innovation.square()*metrics[side][:, None]).sum(-1).gather(1, order)
                    corrected_error = ((target-corrected).square()*metrics[side][:, None]).sum(-1).gather(1, order)
                    for fi, fraction in enumerate(FRACTIONS):
                        count = math.ceil(n*fraction)
                        losses[fi, depth-1] = (corrected_error[:, :count].sum(1)+copy_error[:, count:].sum(1))/n
                assert torch.isfinite(losses).all()
                record[side] = {'anchor': anchor_errors[side].cpu().tolist(),
                                'target': losses.permute(2, 0, 1).cpu().tolist(), 'query': query_errors[side].cpu().tolist()}
            output[ranking][str(gop)] = record
    return output
