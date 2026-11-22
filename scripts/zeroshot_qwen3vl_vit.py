"""Zero-training test of the roadmap §3.4 lever: Qwen3-VL-2B's vision encoder on TEXT-ONLY Qwen3-1.7B.

Variants (no training at all):
  A  original Qwen3-1.7B + Qwen3-VL-2B ViT/merger tokens (1D positions, no DeepStack)
  B  A + DeepStack features added at image positions after LLM layers 0..2 (as Qwen3-VL does)
  C  Qwen3-VL-2B itself (reference ceiling)
A/B borrow Qwen3-VL-2B's trained <|vision_start|>/<|vision_end|> embedding rows (ours are untrained).

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/zeroshot_qwen3vl_vit.py [--n-probe 30 --n-qa 30]
"""

import argparse
import io
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

VS, VE, PAD, IM_END = 151652, 151653, 151655, 151645


class Composite:
    def __init__(self, llm, visual, processor, tok, deepstack: bool):
        self.llm, self.visual, self.proc, self.tok, self.deepstack = llm, visual, processor, tok, deepstack
        self._ds, self._mask = None, None
        if deepstack:
            for i, layer in enumerate(llm.model.layers[:3]):
                layer.register_forward_hook(self._hook(i))

    def _hook(self, i):
        def fn(module, inputs, output):
            if self._ds is None:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] != self._mask.shape[1]:  # decoding steps: no image positions
                return output
            hs = hs.clone()
            hs[self._mask] = hs[self._mask] + self._ds[i].to(hs.dtype)
            return (hs, *output[1:]) if isinstance(output, tuple) else hs
        return fn

    @torch.no_grad()
    def generate(self, image, question, max_new=24):
        px = self.proc.image_processor(images=[image], return_tensors="pt")
        grid = px["image_grid_thw"].cuda()
        out = self.visual(px["pixel_values"].cuda().to(self.visual.dtype), grid_thw=grid)
        vis, ds = out.pooler_output, out.deepstack_features
        n = vis.shape[0]
        user = "<|vision_start|>" + "<|image_pad|>" * n + "<|vision_end|>" + question
        text = self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        ids = torch.tensor([self.tok(text, add_special_tokens=False)["input_ids"]]).cuda()
        emb = self.llm.get_input_embeddings()(ids)
        mask = ids == PAD
        emb = emb.masked_scatter(mask.unsqueeze(-1), vis.to(emb.dtype))
        self._ds, self._mask = (ds if self.deepstack else None), mask
        from transformers import DynamicCache

        cache = DynamicCache()
        o = self.llm(inputs_embeds=emb, past_key_values=cache, use_cache=True)
        self._ds = None
        gen = []
        for _ in range(max_new):
            t = int(o.logits[0, -1].argmax())
            if t == IM_END:
                break
            gen.append(t)
            o = self.llm(input_ids=torch.tensor([[t]]).cuda(), past_key_values=cache, use_cache=True)
        return self.tok.decode(gen, skip_special_tokens=True).strip()


class Hybrid:
    """Qwen3-VL architecture with the text-only Qwen3 LM (exports/hybrid-*), native MRoPE + DeepStack."""

    def __init__(self, path):
        from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

        self.m = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
        self.p = AutoProcessor.from_pretrained(path)
        self.tok = AutoTokenizer.from_pretrained(path)

    @torch.no_grad()
    def generate(self, image, question, max_new=24, thinking=False):
        text = self.tok.apply_chat_template([{"role": "user", "content": "<|vision_start|><|image_pad|><|vision_end|>" + question}],
                                            tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        inputs = self.p(text=[text], images=[image], return_tensors="pt").to("cuda")
        out = self.m.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        return self.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).split("</think>")[-1].strip()


class Ceiling:
    def __init__(self, model, processor):
        self.m, self.p = model, processor

    @torch.no_grad()
    def generate(self, image, question, max_new=24):
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        inputs = self.p.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")
        out = self.m.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        return self.p.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-probe", type=int, default=30)
    ap.add_argument("--n-qa", type=int, default=30, help="per official source")
    ap.add_argument("--only-hybrid", help="evaluate only this hybrid export dir")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

    from eval_s1 import load_eval_samples
    from ocr_probe import WORDS, render
    from vlm.eval.answer_match import RELAXED_NUMERIC, answers_match

    if args.only_hybrid:
        systems = {"D_hybrid": Hybrid(str(ROOT / args.only_hybrid))}
        vl = None
    else:
        vl = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", dtype=torch.bfloat16).cuda().eval()
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    if vl is not None:
        systems = {"C_qwen3vl2b_ceiling": Ceiling(vl, proc)}
    for name, ds in ([] if vl is None else [("A_q3_1b7+vit", False), ("B_q3_1b7+vit+deepstack", True)]):
        llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).cuda().eval()
        with torch.no_grad():  # borrow trained boundary rows from the VL model
            E, Evl = llm.get_input_embeddings().weight, vl.get_input_embeddings().weight
            E[[VS, VE]] = Evl[[VS, VE]]
        systems[name] = Composite(llm, vl.model.visual, proc, tok, ds)

    samples = load_eval_samples(ROOT / "manifests/eval-official", None, 1400)
    by_src = {}
    for s in samples:
        by_src.setdefault(s["meta"]["source"], []).append(s)
    qa = [s for src in sorted(by_src) for s in by_src[src][: args.n_qa]]
    report = {}
    for name, sysm in systems.items():
        rng = random.Random(0)
        probe = {}
        for font_px in (80, 40, 20):
            ok = 0
            for i in range(args.n_probe):
                w = WORDS[i % len(WORDS)]
                ans = sysm.generate(render(w, 384, font_px, rng), "What word is written in this image? Answer with the word only.")
                ok += w.lower() in ans.lower()
            probe[font_px] = round(100 * ok / args.n_probe, 1)
        per = {}
        for s in qa:
            q = s["messages"][0]["content"][-1]["text"] + "\nAnswer the question using a single word or phrase."
            ans = sysm.generate(s["images"][0], q)
            src = s["meta"]["source"]
            per.setdefault(src, []).append(any(answers_match(ans, g, relaxed=src.startswith(("chartqa", "infovqa")))
                                               for g in s["meta"]["golds"]))
        accs = {k: round(100 * sum(v) / len(v), 1) for k, v in per.items()}
        report[name] = {"probe": probe, "qa": accs, "qa_overall": round(sum(sum(v) for v in per.values()) / len(qa) * 100, 1)}
        print(name, json.dumps(report[name]), flush=True)
    out = ROOT / "reports" / ("zeroshot_hybrid.json" if args.only_hybrid else "zeroshot_qwen3vl_vit.json")
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
