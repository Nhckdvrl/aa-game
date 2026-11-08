"""OpenAI-compatible chat server for a Qwen3.5 checkpoint with activation steering (training-free persona control).

Steering modes (vectors from scripts/persona_vectors.py):
  none                 plain model
  resid  --layer L     add alpha * v_L to the residual stream after decoder layer L (all positions)
  heads  --topk K      add alpha * d_h to the input of o_proj for the K full-attention heads whose persona
                       difference is largest relative to their typical activation norm (all positions)

Requests arriving within a few ms are batched into one generate() call (left padding).
Run through the CUDA compat wrapper:
  CUDA_VISIBLE_DEVICES=4 bash /data1/xiangding/cvis-ablate/scripts/h20_env_ablate.sh python scripts/steer_server.py \
      --model CKPT --port 8501 --vectors runs/steer/vectors.pt --mode resid --layer 12 --alpha 2
"""
import argparse
import queue
import threading
import time
import uuid

import torch
import transformers
import uvicorn
from fastapi import FastAPI
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--vectors")
ap.add_argument("--mode", default="none", choices=["none", "resid", "heads"])
ap.add_argument("--layer", type=int, default=12)
ap.add_argument("--topk", type=int, default=4)
ap.add_argument("--alpha", type=float, default=0.0)
ap.add_argument("--decode-only", action="store_true",
                help="steer only newly generated tokens, not the re-encoded conversation (no context contamination)")
ap.add_argument("--mix", default="",
                help="mix several persona vectors: path:layer:alpha,path:layer:alpha (residual stream, overrides --mode)")
ap.add_argument("--max-batch", type=int, default=48)
ap.add_argument("--name", default="ckpt")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda().eval()
layers = model.model.language_model.layers
STEER = {"desc": "none"}

if args.mix:
    add = {}
    for item in args.mix.split(","):
        path, layer, alpha = item.rsplit(":", 2)
        V = torch.load(path)
        add[int(layer)] = add.get(int(layer), 0) + float(alpha) * V["resid"][int(layer)]
    for L, vec in add.items():
        v_mix = vec.to("cuda", torch.bfloat16)

        def mix_hook(mod, inp, out, v_mix=v_mix):
            h = out[0] if isinstance(out, tuple) else out
            if args.decode_only and h.shape[1] != 1:
                return out
            return (h + v_mix,) + tuple(out[1:]) if isinstance(out, tuple) else h + v_mix
        layers[L].register_forward_hook(mix_hook)
    STEER["desc"] = f"mix {args.mix}"
elif args.mode != "none" and args.alpha != 0.0:
    V = torch.load(args.vectors)
    if args.mode == "resid":
        v = (args.alpha * V["resid"][args.layer]).to("cuda", torch.bfloat16)

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if args.decode_only and h.shape[1] != 1:  # prefill of the conversation: leave it unsteered
                return out
            if isinstance(out, tuple):
                return (h + v,) + tuple(out[1:])
            return h + v
        layers[args.layer].register_forward_hook(hook)
        STEER["desc"] = f"resid L{args.layer} alpha={args.alpha}" + (" decode-only" if args.decode_only else "")
    else:
        ranked = V["head_rank"][: args.topk]  # list of (layer, head, score)
        hd = V["head_dim"]
        by_layer = {}
        for L, h, _ in ranked:
            by_layer.setdefault(L, []).append(h)
        for L, hs in by_layer.items():
            add = torch.zeros(V["heads"][L].shape[0] * hd)
            for h in hs:
                add[h * hd:(h + 1) * hd] = args.alpha * V["heads"][L][h]
            add = add.to("cuda", torch.bfloat16)

            def pre(mod, inp, add=add):
                return (inp[0] + add,) + tuple(inp[1:])
            layers[L].self_attn.o_proj.register_forward_pre_hook(pre)
        STEER["desc"] = f"heads top{args.topk} {[(L, h) for L, h, _ in ranked]} alpha={args.alpha}"
print("steering:", STEER["desc"], flush=True)

Q: "queue.Queue" = queue.Queue()


def worker():
    while True:
        batch = [Q.get()]
        t0 = time.time()
        while len(batch) < args.max_batch and time.time() - t0 < 0.02:
            try:
                batch.append(Q.get(timeout=0.005))
            except queue.Empty:
                pass
        texts = [tok.apply_chat_template(b["messages"], add_generation_prompt=True, enable_thinking=False,
                                         tokenize=False) for b in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        max_new = min(320, max(b["max_tokens"] for b in batch))
        temp = batch[0]["temperature"]
        try:
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new, do_sample=temp > 0,
                                     temperature=max(temp, 1e-5), top_p=1.0, pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            for b, g in zip(batch, gen):
                b["result"] = tok.decode(g, skip_special_tokens=True).strip()
                b["event"].set()
        except Exception as e:  # report the error to every waiting request (and log it)
            print('generate failed:', repr(e)[:300], 'batch', len(batch), 'max_len', enc['input_ids'].shape[1], flush=True)
            torch.cuda.empty_cache()
            for b in batch:
                b["error"] = repr(e)
                b["event"].set()


threading.Thread(target=worker, daemon=True).start()
app = FastAPI()


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": args.name, "object": "model", "steer": STEER["desc"]}]}


@app.post("/v1/chat/completions")
def chat(req: dict):
    item = {"messages": req["messages"], "max_tokens": int(req.get("max_tokens") or 200),
            "temperature": float(req.get("temperature", 0.7)), "event": threading.Event()}
    Q.put(item)
    item["event"].wait()
    if "error" in item:
        return {"error": {"message": item["error"]}}
    return {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
            "model": args.name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": item["result"]},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
