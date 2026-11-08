"""Expression shaping for ms-swift GRPO (after Goodfire, "Shaping and Monitoring Models Through Their Internal
Representations", 2026): add  lambda * mean_over_completion_tokens( relu(h_L . v - tau) )  to the GRPO loss, where
h_L is the policy's own residual stream after decoder layer L and v a unit concept direction (here: "about to give
away the whole answer", from a linear probe). Load as an extra plugin:  --external_plugins hds/swift_env.py hds/shaping.py

Env: HDS_SHAPE_VEC (torch file with {"v": [D], "tau": float, "layer": int}), HDS_SHAPE_LAMBDA (default 0.05),
     HDS_SHAPE_LOG (optional jsonl: one line per loss call with the penalty and mean projection),
     HDS_SHAPE_POS = all (mean over every completion token, default) | gen (mean over the first HDS_SHAPE_WIN=6 positions
     of each assistant turn -- the empty think block and the position where the probe was fit, where the decision to
     dump is made). HDS_SHAPE_RELOAD=1 re-reads the direction file whenever it changes on disk, so an outside loop
     can refit the probe on the policy's own recent episodes and swap it in (scripts/refit_loop.sh).
     HDS_SHAPE_REL=<margin> replaces the fixed threshold by (mean projection over the penalized positions of the batch,
     detached) + margin: shifting every projection down no longer helps, only turns that stand out are penalized.

Swift runs DeepSpeed with *reentrant* gradient checkpointing, so a hook on layer L itself sees no-grad tensors. Layer
L+1's forward-pre-hook receives the output of layer L's checkpoint wrapper, which is part of the autograd graph -- but only
if it runs before DeepSpeed ZeRO-3's own pre-hooks: those run inside the checkpoint under no_grad and pass on detached
copies (hence prepend=True).
"""
import json
import os
import time

import torch

VEC = os.environ.get("HDS_SHAPE_VEC")
LAM = float(os.environ.get("HDS_SHAPE_LAMBDA", "0.05"))
LOG = os.environ.get("HDS_SHAPE_LOG")
POS = os.environ.get("HDS_SHAPE_POS", "all")  # all: every completion token; gen: first WIN positions of each assistant turn
WIN = int(os.environ.get("HDS_SHAPE_WIN", "6"))  # covers the empty <think></think> block the policy emits before replying
RELOAD = os.environ.get("HDS_SHAPE_RELOAD") == "1"  # re-read HDS_SHAPE_VEC when the file changes (refit monitors)
REL = os.environ.get("HDS_SHAPE_REL")  # if set (a margin): threshold = batch mean projection + margin, so an offset cannot dodge it

if VEC:
    from swift.rlhf_trainers import grpo_trainer as _g

    _spec = torch.load(VEC)
    _v, _tau, _layer = _spec["v"].float(), float(_spec["tau"]), int(_spec["layer"])
    _store = {}
    _orig = _g.GRPOTrainer._compute_loss_and_metrics
    _ver = {"mtime": os.path.getmtime(VEC), "n": 0}

    def _maybe_reload():
        global _v, _tau
        try:
            m = os.path.getmtime(VEC)
            if m == _ver["mtime"]:
                return
            spec = torch.load(VEC)
        except Exception:  # missing or half-written: keep the current direction, try again next call
            return
        _v, _tau = spec["v"].float(), float(spec["tau"])
        _ver["mtime"], _ver["n"] = m, _ver["n"] + 1
        if LOG and int(os.environ.get("RANK", "0")) == 0:
            with open(LOG, "a") as f:
                f.write(json.dumps({"t": time.time(), "reload": _ver["n"], "tau": _tau, "src": str(spec.get("src", ""))}) + "\n")

    def _find_layers(m):
        for path in ("model.language_model.layers", "model.model.language_model.layers", "model.layers"):
            obj = m
            try:
                for p in path.split("."):
                    obj = getattr(obj, p)
                return obj
            except AttributeError:
                continue
        raise RuntimeError("decoder layers not found")

    def _pre(mod, args):
        _store["h"] = args[0]

    _dbg = {"n": 0}

    def _debug(msg):
        if LOG and int(os.environ.get("RANK", "0")) == 0 and _dbg["n"] < 6:
            with open(LOG + ".debug", "a") as f:
                f.write(msg + "\n")

    def _patched(self, model, model_inputs, grpo_batch):
        if RELOAD:
            _maybe_reload()
        if not getattr(self, "_hds_shape_hooked", False):
            layers = _find_layers(self.model)
            layers[_layer + 1].register_forward_pre_hook(_pre, prepend=True)  # before DeepSpeed ZeRO-3 hooks, which detach under no_grad
            self._hds_shape_hooked = True
            _debug(f"hooked layer {_layer + 1} of {len(layers)} on {type(self.model).__name__}")
        _store.clear()
        loss, metrics = _orig(self, model, model_inputs, grpo_batch)
        h = _store.get("h")
        _debug(f"call {_dbg['n']}: h={None if h is None else tuple(h.shape)} grad={None if h is None else h.requires_grad} "
               f"training={self.model.training} k={getattr(grpo_batch, 'logits_to_keep', None)} "
               f"mask={tuple(grpo_batch.completion_mask.shape)} grad_enabled={torch.is_grad_enabled()}")
        _dbg["n"] += 1
        if h is not None and self.model.training and h.requires_grad:
            k = int(grpo_batch.logits_to_keep)
            mask = grpo_batch.completion_mask.float()
            if POS == "gen":  # turn starts: mask switches 0 -> 1 (or the first completion position), then WIN positions
                prev = torch.cat([torch.zeros_like(mask[:, :1]), mask[:, :-1]], dim=1)
                start = mask * (1 - prev)
                win = start.clone()
                for d in range(1, WIN):
                    win = win + torch.nn.functional.pad(start, (d, 0))[:, :-d]
                mask = mask * win.clamp(max=1.0)
            hh = h[:, -k - 1:-1, :].float()
            proj = hh @ _v.to(hh.device)
            if REL is not None:
                thr = ((proj * mask).sum() / mask.sum().clamp(min=1.0)).detach() + float(REL)
            else:
                thr = _tau
            pen = (torch.relu(proj - thr) * mask).sum() / mask.sum().clamp(min=1.0)
            loss = loss + LAM * pen
            if LOG and int(os.environ.get("RANK", "0")) == 0:
                with open(LOG, "a") as f:
                    f.write(json.dumps({"t": time.time(), "penalty": float(pen.detach()),
                                        "proj_mean": float((proj * mask).sum().detach() / mask.sum().clamp(min=1.0)),
                                        "grad": bool(pen.requires_grad), "n": int(mask.sum()),
                                        "proj_max": float((proj - 1e9 * (1 - mask)).max().detach())}) + "\n")
        return loss, metrics

    _g.GRPOTrainer._compute_loss_and_metrics = _patched
