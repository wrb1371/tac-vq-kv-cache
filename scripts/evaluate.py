"""Portable native video evaluation for the released STI-RVQ implementation."""

import argparse, collections, gc, json, os, pathlib, re, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/OmniKVQuant_snapshot")]


def load_model(backbone, model_dir):
    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    original = torch.cuda.set_per_process_memory_fraction
    limit = min(0.95, 26 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.set_per_process_memory_fraction = lambda fraction, device=None: original(
        min(fraction, limit), device
    )
    if backbone == "vl7b":
        import ar26kv.runtime as runtime

        runtime.MODEL = pathlib.Path(model_dir)
        model, p = runtime.load()
    else:
        from transformers import Qwen2_5OmniConfig
        from model.modeling_qwen2_5_omni import load_backbone
        from inference.inputs import seed_everything
        from ar20kv.attention import install_reference_attention
        from r13kv.media_cache import install_media_cache

        seed_everything(42)
        cfg = Qwen2_5OmniConfig.from_pretrained(model_dir, local_files_only=True)
        model, p = load_backbone(
            model_dir, config=cfg, attn_implementation="flash_attention_2"
        )
        model.requires_grad_(False)
        install_media_cache()
        install_reference_attention(model, "gqa")
    torch.cuda.set_per_process_memory_fraction = original
    return model, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["omni3b", "vl7b"], required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--method",
        choices=[
            "sti-rvq",
            "bf16",
            "snapkv",
            "anchorkv",
            "pyramidkv",
            "expected-attention",
            "infinipot-v",
            "cords",
            "gsrq",
            "nova-kv",
        ],
        default="sti-rvq",
    )
    ap.add_argument(
        "--baseline-root",
        help="External baseline bank root, preserving the documented artifacts/ layout",
    )
    ap.add_argument(
        "--reference-reader",
        action="store_true",
        help="Use the independent PyTorch reader for VL-7B",
    )
    ap.add_argument(
        "--save-state-dir",
        help="Optional packed VL STI-RVQ snapshots for reader verification/benchmarking",
    )
    a = ap.parse_args()
    if a.save_state_dir:
        assert a.backbone == "vl7b" and a.method == "sti-rvq"

    if a.baseline_root:
        os.environ["STIRVQ_BASELINE_ROOT"] = str(
            pathlib.Path(a.baseline_root).resolve()
        )
    import torch
    from ar32kv.runtime import Supplier, answer

    model, p = load_model(a.backbone, a.model_dir)
    kind = "VL" if a.backbone == "vl7b" else "Omni"
    supplier = Supplier(model, p, kind)
    items = json.loads(pathlib.Path(a.manifest).read_text())
    assert items and len({x["id"] for x in items}) == len(items)
    groups = collections.OrderedDict()
    for x in items:
        groups.setdefault(x["video"], []).append(x)
    path = pathlib.Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = correct = infeasible = 0
    start = time.time()
    with path.open("x") as stream:
        for video, questions in groups.items():
            x, pos, vi, ti, end, identity = supplier.get(questions[0], verify=True)
            codec = None
            teacher = None
            native_cache = None
            try:
                if a.method in [
                    "snapkv",
                    "anchorkv",
                    "pyramidkv",
                    "expected-attention",
                ]:
                    names = {
                        "snapkv": "SnapKV",
                        "anchorkv": "AnchorKV",
                        "pyramidkv": "PyramidKV",
                        "expected-attention": "ExpectedAttention",
                    }
                    if a.method in ["pyramidkv", "expected-attention"]:
                        from ar34kv.runtime import Teacher
                        from ar34kv.codec import Codec
                    else:
                        from ar32kv.runtime import Teacher
                        from ar32kv.codec import Codec
                    teacher = Teacher(model, kind)
                    teacher.capture(x, pos, end)
                    codec = Codec(names[a.method], questions[0], vi, ti, teacher)
                elif kind == "Omni" and a.method == "bf16":
                    from ar32kv.runtime import Teacher

                    teacher = Teacher(model, kind)
                    teacher.capture(x, pos, end)
                elif kind == "VL":
                    from ar26kv.runtime import Teacher
                    from ar26kv.codec import Codec

                    teacher = Teacher(model)
                    model._ar26_grid = identity["grid"]
                    teacher.capture(x, pos, end)
                    if a.method != "bf16":
                        if a.method == "gsrq":
                            if questions[0]["dataset"] == "TempCompass-MC":
                                codec = Codec(
                                    "GSRQ-export",
                                    model,
                                    questions[0],
                                    pos,
                                    vi,
                                    ti,
                                    end,
                                    teacher,
                                )
                            else:
                                from ar31kv.codec import BudgetGSRQ

                                codec = BudgetGSRQ(
                                    "GSRQ-export",
                                    model,
                                    questions[0],
                                    pos,
                                    vi,
                                    ti,
                                    end,
                                    teacher,
                                )
                        elif a.method == "nova-kv":
                            from ar30kv.codec import NativeControl

                            codec = NativeControl(
                                "NOVA-KV-g8",
                                model,
                                questions[0],
                                pos,
                                vi,
                                ti,
                                end,
                                teacher,
                            )
                        else:
                            name = {
                                "sti-rvq": "STI-RVQ",
                                "infinipot-v": "InfiniPot-V",
                                "cords": "CoRDS",
                            }[a.method]
                            codec = Codec(
                                name, model, questions[0], pos, vi, ti, end, teacher
                            )
                        if a.method == "sti-rvq" and not a.reference_reader:
                            from ar35kv.reader import install

                            install(codec)
                else:
                    from ar22kv.evaluation import make_cache
                    from ar25kv.readonly_v1 import install_scoring_operation
                    from ar18kv.prefix import media_prefill

                    install_scoring_operation()
                    if a.method == "bf16":
                        spec = {"family": "BF16"}
                    else:
                        name = {
                            "sti-rvq": "A25_S000_RO",
                            "infinipot-v": "InfiniPot",
                            "cords": "CoRDS",
                            "gsrq": "GSRQTier",
                            "nova-kv": "NOVA8",
                        }[a.method]
                        spec = json.loads(
                            (ROOT / "configs/ar22" / (name + ".json")).read_text()
                        )
                        if a.baseline_root and a.method in ["gsrq", "nova-kv"]:
                            spec["package"] = str(
                                pathlib.Path(a.baseline_root) / spec["package"]
                            )
                            if "package_by_budget" in spec.get("policy", {}):
                                spec["policy"]["package_by_budget"] = {
                                    k: str(pathlib.Path(a.baseline_root) / v)
                                    for k, v in spec["policy"][
                                        "package_by_budget"
                                    ].items()
                                }
                    spec.setdefault("policy", {})["read_only_single_step"] = True
                    codec, native_cache = make_cache(
                        model, spec, questions[0], x, pos, identity, end
                    )
                    media_prefill(model, x, native_cache, pos, end, False)
            except RuntimeError as exc:
                if not any(w in str(exc) for w in ["INFEASIBLE", "NOT_FEASIBLE"]):
                    raise
                for it in questions:
                    stream.write(
                        json.dumps(
                            {
                                "id": it["id"],
                                "source_id": it["source_id"],
                                "dataset": it["dataset"],
                                "method": a.method,
                                "backbone": a.backbone,
                                "status": "INFEASIBLE",
                                "correct": None,
                                "reason": str(exc),
                            }
                        )
                        + "\n"
                    )
                    count += 1
                    infeasible += 1
                stream.flush()
                if teacher:
                    teacher.close()
                del codec, teacher, native_cache, x, pos, vi, ti
                gc.collect()
                torch.cuda.empty_cache()
                continue
            if a.save_state_dir:
                dest = pathlib.Path(a.save_state_dir)
                dest.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "states": {
                            l: {k: v.cpu() for k, v in st.items() if torch.is_tensor(v)}
                            for l, st in codec.states.items()
                        },
                        "table": {k: v.cpu() for k, v in codec.table.items()},
                        "global_maps": {
                            k: v.cpu() for k, v in codec.global_maps.items()
                        },
                        "plan": codec.plan,
                        "grid": codec.grid,
                        "scale": codec.scale,
                        "end": codec.end,
                        "receipt": codec.receipt,
                    },
                    dest / (pathlib.Path(video).stem + ".pt"),
                )
            for it in questions:
                xx, pp, vv, tt, ee, ident = supplier.get(it)
                assert ee == end
                if teacher is not None:
                    teacher.assert_same(xx, pp, ee)
                if native_cache is not None:
                    from ar22kv.evaluation import answer as score

                    pred = score(model, p, native_cache, xx, pp, end, it)
                    fee = native_cache.receipt() if codec else None
                else:
                    cache = codec.cache() if codec else teacher.bf16_cache()
                    pred = answer(model, p, kind, xx, pp, end, cache, it)
                    fee = codec.receipt if codec else None
                    del cache
                text = p.tokenizer.decode(
                    pred.get("token_ids", [pred.get("generated_token_id")]),
                    skip_special_tokens=True,
                ).strip()
                match = re.search(r"\b([A-H])\b", text)
                choice = match.group(1) if match else None
                row = {
                    "id": it["id"],
                    "source_id": it["source_id"],
                    "dataset": it["dataset"],
                    "duration_group": it["duration_group"],
                    "backbone": a.backbone,
                    "method": a.method,
                    "status": "COMPLETE",
                    "input_identity": ident,
                    "budget_mib": it["budget_mib"],
                    "receipt": fee,
                    "prediction": choice,
                    "correct": choice == it["answer"],
                    **pred,
                }
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                count += 1
                correct += row["correct"]
                del xx, pp, vv, tt
            if teacher:
                teacher.close()
            if native_cache is not None and codec:
                native_cache.codec.finish_request()
            del codec, teacher, native_cache, x, pos, vi, ti
            gc.collect()
            torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "completed": count,
                        "accuracy_pct": None if infeasible else 100 * correct / count,
                        "elapsed_s": time.time() - start,
                    }
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "complete": True,
                "questions": count,
                "infeasible": infeasible,
                "correct": correct,
                "accuracy_pct": None if infeasible else 100 * correct / count,
                "output": str(path),
            }
        )
    )


if __name__ == "__main__":
    main()
