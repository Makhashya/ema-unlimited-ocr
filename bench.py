"""Benchmark: looping infer() per image vs one infer_multi() over all images."""
import contextlib, io, json, os, time
import torch
from transformers import AutoModel, AutoTokenizer

PAGES = [f"examples/p{i:02d}.png" for i in range(1, 9)]
OUT = "bench_out"

tok = AutoTokenizer.from_pretrained("baidu/Unlimited-OCR", trust_remote_code=True)
model = AutoModel.from_pretrained("baidu/Unlimited-OCR", trust_remote_code=True,
                                  use_safetensors=True, dtype=torch.bfloat16).eval().cuda()
base = torch.cuda.memory_allocated()
print(f"weights resident: {base/1e9:.2f} GB")

def run(label, fn):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    dt = time.perf_counter() - t
    peak = torch.cuda.max_memory_allocated()
    chars = len(buf.getvalue())
    print(json.dumps({"case": label, "sec": round(dt, 1),
                      "peak_VRAM_GB": round(peak/1e9, 2),
                      "above_weights_GB": round((peak-base)/1e9, 2),
                      "out_chars": chars}))
    return dt

def loop(pages, mode):
    gundam = mode == "gundam"
    def f():
        for p in pages:
            model.infer(tok, prompt="<image>document parsing.", image_file=p,
                        output_path=OUT, base_size=1024,
                        image_size=640 if gundam else 1024, crop_mode=gundam,
                        max_length=32768, no_repeat_ngram_size=35, ngram_window=128,
                        save_results=False)
    return f

def multi(pages):
    def f():
        model.infer_multi(tok, prompt="<image>Multi page parsing.", image_files=pages,
                          output_path=OUT, image_size=1024, max_length=32768,
                          no_repeat_ngram_size=35, ngram_window=1024, save_results=False)
    return f

for n in (4, 8):
    run(f"loop infer() gundam x{n}", loop(PAGES[:n], "gundam"))
    run(f"loop infer() base   x{n}", loop(PAGES[:n], "base"))
    run(f"infer_multi()       x{n}", multi(PAGES[:n]))
