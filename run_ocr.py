"""Transformers-path runner for Unlimited-OCR (Windows / single NVIDIA GPU).

The repo's own infer.py drives an SGLang server, which has no Windows wheels.
This wraps the HuggingFace transformers API documented in README.md instead.

    python run_ocr.py --image page.jpg --output_dir outputs
    python run_ocr.py --pdf report.pdf --output_dir outputs
    python run_ocr.py --images p1.png p2.png --output_dir outputs
"""

import argparse
import os
import tempfile

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = os.environ.get("UNLIMITED_OCR_MODEL", "baidu/Unlimited-OCR")


def pdf_to_images(pdf_path, dpi=300):
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    return tokenizer, model.eval().cuda()


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="single image -> gundam or base mode")
    src.add_argument("--images", nargs="+", help="multiple pages -> base mode")
    src.add_argument("--pdf", help="PDF -> rendered to pages, base mode")
    src.add_argument("--image_dir", help="folder of INDEPENDENT images -> loops infer(), "
                                         "model loaded once, one result file per image")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--mode", choices=["gundam", "base"], default="gundam",
                    help="single-image only; multi-page always uses base")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer, model = load_model()

    if args.image_dir:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        files = sorted(f for f in os.listdir(args.image_dir)
                       if os.path.splitext(f)[1].lower() in exts)
        if not files:
            raise SystemExit(f"no images found in {args.image_dir}")
        gundam = args.mode == "gundam"
        for i, name in enumerate(files, 1):
            sub = os.path.join(args.output_dir, os.path.splitext(name)[0])
            os.makedirs(sub, exist_ok=True)
            print(f"[{i}/{len(files)}] {name}")
            model.infer(
                tokenizer,
                prompt="<image>document parsing.",
                image_file=os.path.join(args.image_dir, name),
                output_path=sub,
                base_size=1024,
                image_size=640 if gundam else 1024,
                crop_mode=gundam,
                max_length=32768,
                no_repeat_ngram_size=35,
                ngram_window=128,
                save_results=True,
            )
    elif args.image:
        # gundam: base_size=1024, image_size=640, crop_mode=True
        # base:   base_size=1024, image_size=1024, crop_mode=False
        gundam = args.mode == "gundam"
        model.infer(
            tokenizer,
            prompt="<image>document parsing.",
            image_file=args.image,
            output_path=args.output_dir,
            base_size=1024,
            image_size=640 if gundam else 1024,
            crop_mode=gundam,
            max_length=32768,
            no_repeat_ngram_size=35,
            ngram_window=128,
            save_results=True,
        )
    else:
        pages = args.images if args.images else pdf_to_images(args.pdf, dpi=args.dpi)
        model.infer_multi(
            tokenizer,
            prompt="<image>Multi page parsing.",
            image_files=pages,
            output_path=args.output_dir,
            image_size=1024,
            max_length=32768,
            no_repeat_ngram_size=35,
            ngram_window=1024,
            save_results=True,
        )

    print(f"\nResults written to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
