#!/usr/bin/env python
"""
Batch-convert PDFs to text with Docling (RTMDet-S layout + RapidOCR).

Usage
-----
# One file
python batch_docling.py /path/to/file.pdf

# Folder of PDFs (non-recursive)
python batch_docling.py /path/to/folder
"""
from __future__ import annotations
import argparse
import pathlib
import statistics
import sys
import time
import logging
from tqdm import tqdm
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption

# ────────────── enable Docling’s per-page logs ──────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s:%(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
# optional: silence noisy sub-modules
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# ────────────── helper ──────────────
def convert_one(pdf_path: pathlib.Path):
    accel = AcceleratorOptions(device=AcceleratorDevice.CUDA)
    pipe = PdfPipelineOptions(
        accelerator_options=accel,
        do_ocr=True,
        ocr_options=RapidOcrOptions(lang=["en"]),
        do_table_structure=True,
        layout_model_spec="mmdet_rtm_s",
    )
    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipe)}
    )

    t0 = time.perf_counter()
    res = conv.convert(str(pdf_path))          # blocks until PDF done
    total = time.perf_counter() - t0

    per_page = (
        res.timings["page_total"].times
        if res.timings.get("page_total") and res.timings["page_total"].times
        else [total / res.document.num_pages()] * res.document.num_pages()
    )
    txt = res.document.export_to_text()
    return txt, total, per_page

# ────────────── main ──────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="PDF file or directory")
    args = ap.parse_args()

    target = pathlib.Path(args.path).expanduser()
    if not target.exists():
        sys.exit("Path not found")

    pdfs = [target] if target.is_file() else sorted(target.glob("*.pdf"))
    if not pdfs:
        sys.exit("No PDF files found")

    grand_total, all_pages = 0.0, []
    print(f"▶ Processing {len(pdfs)} PDF(s)…")

    for pdf in tqdm(pdfs, unit="pdf", colour="cyan"):
        txt, tot, per = convert_one(pdf)
        grand_total += tot
        all_pages.extend(per)

        out_dir = pdf.with_suffix("")
        out_dir.mkdir(exist_ok=True)
        (out_dir / "docling.txt").write_text(txt, encoding="utf-8")

        tqdm.write(
            f"{pdf.name:40s} | pages {len(per):3d} | {tot:6.2f}s "
            f"(mean {statistics.mean(per):.3f}s)"
        )

    print(
        "\n=== summary ===============================\n"
        f" PDFs processed : {len(pdfs)}\n"
        f" Pages total    : {len(all_pages)}\n"
        f" Total time     : {grand_total:6.2f}s\n"
        f" Per-page (s)   : "
        f"min {min(all_pages):.3f} | mean {statistics.mean(all_pages):.3f} | "
        f"max {max(all_pages):.3f}\n"
    )

if __name__ == "__main__":
    main()
