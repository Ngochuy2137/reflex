"""Stage 3, step 2: wrap a trained heatmap CNN into a feature-extractor checkpoint for feature collection.

Feature collection runs reflex.eval.per_throw_eval with the Stage 2 controller driving on its privileged latent
(--det_shadow) and a visual encoder alongside it (--detector); --save_feats logs, per control step, the encoder's
per-frame features DetectorKPSeq.frame_feats (8-D: heatmap soft-argmax, local-min depth, visibility logit, pinhole
lift) plus the camera ego-motion (9-D), with the controller's latent z* as the label. Those features depend only on
the CNN, the heatmap head and the spatial softmax of the encoder, i.e. on the train_detector.py checkpoint.
The evaluator loads a complete 'kpseq' checkpoint, so this script builds one: the cnn / heat / ssm tensors are
copied from --cnn, the GRU and output heads are freshly initialised (seeded). They only produce the diagnostic
z_det of the shadow file and never enter the saved features.

usage: python -m reflex.vision.make_feature_net --cnn data/stage3/<run>/heatmap_cnn.pt --out data/stage3/<run>/feature_net.pt
"""
import argparse
import os

import torch

from reflex.vision.detector_model import DetectorKPSeq

FRONT_END = ("cnn.", "heat.", "ssm.")   # the sub-modules of DetectorKPSeq.kp that frame_feats uses


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cnn", required=True, help="heatmap CNN from reflex.vision.train_detector (arch kp / kpm), or a kpseq checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--win", type=int, default=25, help="frame history of the evaluator's feature ring")
    p.add_argument("--zdim", type=int, default=8, help="latent size of the (unused) GRU head; 8 = the Stage 2 latent")
    p.add_argument("--seed", type=int, default=0, help="seed of the GRU / head initialisation")
    a = p.parse_args()
    torch.manual_seed(a.seed)

    ck = torch.load(a.cnn, map_location="cpu", weights_only=False)
    arch, in_ch = ck.get("arch"), int(ck.get("in_ch", 3))
    if arch not in ("kp", "kpm", "kpseq") or in_ch != 4:
        raise SystemExit(f"[featnet] {a.cnn}: need an RGB-D heatmap checkpoint (arch kp / kpm / kpseq, in_ch 4), got arch {arch} in_ch {in_ch}")
    src = {(k[3:] if arch == "kpseq" else k): v for k, v in ck["model"].items() if arch != "kpseq" or k.startswith("kp.")}
    src = {k: v for k, v in src.items() if k.startswith(FRONT_END)}

    model = DetectorKPSeq(a.win, a.zdim, in_ch=4)
    want = {k: v for k, v in model.kp.state_dict().items() if k.startswith(FRONT_END)}
    missing = sorted(set(want) - set(src))
    bad = sorted(k for k in want if k in src and tuple(src[k].shape) != tuple(want[k].shape))
    if missing or bad:
        raise SystemExit(f"[featnet] {a.cnn} does not match DetectorKPSeq: missing {missing}, shape mismatch {bad}")
    model.kp.load_state_dict({k: src[k] for k in want}, strict=False)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"model": model.state_dict(), "arch": "kpseq", "in_ch": 4, "win": a.win, "zdim": a.zdim, "hidden": 128,
                "dropout": 0.0, "kp_src": os.path.basename(a.cnn)}, a.out)
    print(f"[featnet] copied {len(want)} front-end tensors ({', '.join(FRONT_END)}) from {a.cnn} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
