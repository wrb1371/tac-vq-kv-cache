"""Verify every reconstructed BF16 element on a supplied packed VL-7B state."""

import argparse, pathlib, sys, json

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/OmniKVQuant_snapshot")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    a = ap.parse_args()
    import torch
    from ar26kv.codec import Codec
    from ar35kv.reader import install

    torch.backends.cuda.matmul.allow_tf32 = False
    raw = torch.load(a.state, map_location="cpu", weights_only=True)
    c = object.__new__(Codec)
    for k in ["plan", "grid", "scale", "end", "receipt"]:
        setattr(c, k, raw[k])
    c.name = "STI-RVQ"
    c.states = {
        l: {k: v.cuda() for k, v in st.items()} for l, st in raw["states"].items()
    }
    c.table = {k: v.cuda() for k, v in raw["table"].items()}
    c.global_maps = {k: v.cuda() for k, v in raw["global_maps"].items()}
    install(c)
    for l in range(28):
        expected = c._ar35_reference_decode(l)
        actual = c.decode(l)
        assert all(torch.equal(x, y) for x, y in zip(expected, actual)), l
    print(
        json.dumps(
            {
                "all28layers_BF16_exact": True,
                "resident_bytes": c.receipt["resident_bytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
