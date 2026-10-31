"""Independent private Fisher-CQ port, pre-MRoPE keys and exact packed capacity."""

import json, os, pathlib, torch
from transformers.cache_utils import DynamicCache
from r15kv.persistent_registry import PersistentRegistry
from ar26kv.codec import apply_rope
from .admission import cost
from .fitting import nearest
from .packing import pack, unpack
from .runtime import ROOT, geometry


class ReadCache(DynamicCache):
    def __init__(self, c):
        super().__init__()
        self.codec = c
        self._seen_tokens = c.end

    def get_seq_length(self, layer_idx=0):
        return self.codec.end

    def update(self, k, v, layer_idx, cache_kwargs=None):
        a, b = self.codec.decode(layer_idx)
        return torch.cat([a[None], k], 2), torch.cat([b[None], v], 2)


class Codec:
    @torch.inference_mode()
    def __init__(self, model, kind, item, pos, vi, ti, end, teacher, C, c, gate=False):
        language, (self.L, _, self.H, self.D) = geometry(model, kind)
        self.end = end
        self.C = C
        self.c = c
        self.J = self.D // c
        self.b = C.bit_length() - 1
        self.M = len(vi)
        self.T = len(ti)
        self.Q = item["query_reservation_tokens"]
        self.W = min(32, self.M)
        self.N = self.M - self.W
        self.slots = self.N + self.Q
        self.plan = cost(kind, C, c, self.M, self.T, self.Q)
        self.registry = PersistentRegistry(int(item["budget_mib"] * 2**20))
        self.states = {}
        self.books = {}
        folder = (
            pathlib.Path(os.environ.get("STIRVQ_BASELINE_ROOT", str(ROOT)))
            / f"artifacts/ar37/packages/{kind}/C{C}_c{c}"
        )
        complete = json.loads((folder / "completion.json").read_text())
        assert complete["completed"]
        for l in range(self.L):
            self.books[l] = {
                s: v.cuda()
                for s, v in torch.load(
                    folder / f"layer{l:02}.pt", weights_only=True
                ).items()
            }
        self.registry.own("codec", self.books, "codec")
        coords = torch.zeros((3, end + self.Q), device="cuda", dtype=torch.int32)
        coords[:, :end] = pos[:, :end] if kind == "Omni" else pos[:, 0, :end]
        self.maps = {
            "positions": coords,
            "coded": vi[: self.N].to(torch.int32).clone(),
            "protected": torch.cat([ti, vi[self.N :]]).to(torch.int32),
            "freq": language.rotary_emb.inv_freq.detach().float().clone(),
        }
        self.scale = float(language.rotary_emb.attention_scaling)
        self.registry.own("cache/maps", self.maps)
        for l in range(self.L):
            st = {}
            raw = teacher.states[l]
            for s in ["K", "V"]:
                z = self.books[l][s].float().reshape(self.H * self.J, C, c)
                xx = (
                    raw["K_pre" if s == "K" else "V"][:, vi[: self.N].cpu()]
                    .float()
                    .reshape(self.H, self.N, self.J, c)
                    .permute(0, 2, 1, 3)
                    .reshape(self.H * self.J, self.N, c)
                    .cuda()
                )
                ix = nearest(xx, z).reshape(self.H, self.J, self.N).permute(0, 2, 1)
                full = torch.zeros(
                    (self.H, self.slots, self.J), dtype=torch.long, device="cuda"
                )
                full[:, : self.N] = ix
                st[s + "_index"] = pack(full, self.b)
                st[s + "_protected"] = (
                    raw["K_post" if s == "K" else "V"][
                        :, self.maps["protected"].cpu().long()
                    ]
                    .cuda()
                    .clone()
                )
                if gate:
                    assert torch.equal(
                        unpack(st[s + "_index"], self.b, full.numel()).reshape_as(full),
                        full,
                    )
            self.states[l] = st
            self.registry.own(f"cache/layer{l}", st)
        self.receipt = self.registry.receipt()
        assert self.receipt["resident_bytes"] == self.plan["resident_bytes"], (
            self.receipt,
            self.plan,
        )

    def decode_side(self, l, s):
        ids = unpack(
            self.states[l][s + "_index"], self.b, self.H * self.slots * self.J
        ).reshape(self.H, self.slots, self.J)[:, : self.N]
        book = self.books[l][s]
        hi = torch.arange(self.H, device="cuda")[:, None, None]
        ji = torch.arange(self.J, device="cuda")[None, None, :]
        vals = book[hi, ji, ids].reshape(self.H, self.N, self.D).bfloat16()
        out = torch.zeros(
            (self.H, self.end, self.D), device="cuda", dtype=torch.bfloat16
        )
        out[:, self.maps["coded"].long()] = vals
        if s == "K":
            out = apply_rope(
                out,
                self.maps["positions"][:, : self.end],
                self.maps["freq"],
                self.scale,
            )
        out[:, self.maps["protected"].long()] = self.states[l][s + "_protected"]
        return out

    def decode(self, l):
        return self.decode_side(l, "K"), self.decode_side(l, "V")

    def cache(self):
        return ReadCache(self)
