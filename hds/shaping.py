"""Expression shaping for ms-swift GRPO (after Goodfire, "Shaping and Monitoring Models Through Their Internal
Representations", 2026): add  lambda * mean_over_completion_tokens( relu(h_L . v - tau) )  to the GRPO loss, where
h_L is the policy's own residual stream after decoder layer L and v a unit concept direction (here: "about to give
away the whole answer", from a linear probe). Load as an extra plugin:  --external_plugins hds/swift_env.py hds/shaping.py

Env: HDS_SHAPE_VEC (torch file with {"v": [D], "tau": float, "layer": int}), HDS_SHAPE_LAMBDA (default 0.05),
     HDS_SHAPE_LOG (optional jsonl: one line per loss call with the penalty and mean projection).

Swift runs DeepSpeed with *reentrant* gradient checkpointing, so a hook on layer L itself sees no-grad tensors. Layer
L+1's forward-pre-hook receives the output of layer L's checkpoint wrapper, which is part of the autograd graph.
"""
import json
import os
import time

import torch

VEC = os.environ.get("HDS_SHAPE_VEC")
LAM = float(os.environ.get("HDS_SHAPE_LAMBDA", "0.05"))
LOG = os.environ.get("HDS_SHAPE_LOG")

if VEC:
    from swift.rlhf_trainers import grpo_trainer as _g

    _spec = torch.load(VEC)
    _v, _tau, _layer = _spec["v"].float(), float(_spec["tau"]), int(_spec["layer"])
    _store = {}
    _orig = _g.GRPOTrainer._compute_loss_and_metrics

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

    def _patched(self, model, model_inputs, grpo_batch):
        if not getattr(self, "_hds_shape_hooked", False):
            _find_layers(self.model)[_layer + 1].register_forward_pre_hook(_pre)
            self._hds_shape_hooked = True
        _store.clear()
        loss, metrics = _orig(self, model, model_inputs, grpo_batch)
        h = _store.get("h")
        if h is not None and self.model.training and h.requires_grad:
            k = int(grpo_batch.logits_to_keep)
            mask = grpo_batch.completion_mask.float()
            hh = h[:, -k - 1:-1, :].float()
            proj = hh @ _v.to(hh.device)
            pen = (torch.relu(proj - _tau) * mask).sum() / mask.sum().clamp(min=1.0)
            loss = loss + LAM * pen
            if LOG and int(os.environ.get("RANK", "0")) == 0:
                with open(LOG, "a") as f:
                    f.write(json.dumps({"t": time.time(), "penalty": float(pen.detach()),
                                        "proj_mean": float((proj * mask).sum().detach() / mask.sum().clamp(min=1.0)),
                                        "grad": bool(pen.requires_grad)}) + "\n")
        return loss, metrics

    _g.GRPOTrainer._compute_loss_and_metrics = _patched
