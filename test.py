
#!/usr/bin/env python
# compare_extract.py
import argparse, time, statistics, pathlib
from collections import defaultdict

from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf
import pdfplumber


def extract_with_unstructured(pdf_path: pathlib.Path):
    start = time.perf_counter()
    elements = partition_pdf(filename=str(pdf_path))          # default cascade
    elapsed = time.perf_counter() - start

    text_per_page = defaultdict(list)
    for el in elements:
        if hasattr(el, "text") and el.text:
            text_per_page[el.metadata.page_number].append(el.text)

    # join page texts
    pages = []
    for page_num in sorted(text_per_page):
        pages.append("\n".join(text_per_page[page_num]))
    return pages, elapsed


def extract_with_pdfplumber(pdf_path: pathlib.Path):
    pages = []
    t_per_page = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in tqdm(pdf.pages, desc="pdfplumber", unit="page"):
            t0 = time.perf_counter()
            pages.append(page.extract_text() or "")
            t_per_page.append(time.perf_counter() - t0)
    return pages, sum(t_per_page), t_per_page


def stats(label, total_time, per_page_times):
    print(f"\n=== {label} ===")
    print(f" Pages processed : {len(per_page_times)}")
    print(f" Total time      : {total_time:6.2f} s")
    print(
        " Per-page (s)    : "
        f"min {min(per_page_times):.3f} | "
        f"mean {statistics.mean(per_page_times):.3f} | "
        f"max {max(per_page_times):.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_file", type=pathlib.Path, help="Path to PDF file")
    args = parser.parse_args()

    if not args.pdf_file.exists():
        raise FileNotFoundError(args.pdf_file)

    print("### Extracting with Unstructured …")
    us_pages, us_total = extract_with_unstructured(args.pdf_file)
    us_per_page = [us_total / len(us_pages)] * len(us_pages)  # coarse estimate

    print("### Extracting with pdfplumber …")
    pl_pages, pl_total, pl_per_page = extract_with_pdfplumber(args.pdf_file)

    # save outputs
    out_dir = args.pdf_file.with_suffix("")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "unstructured.txt").write_text("\n\n".join(us_pages), encoding="utf-8")
    (out_dir / "pdfplumber.txt").write_text("\n\n".join(pl_pages), encoding="utf-8")

    # show stats
    stats("Unstructured", us_total, us_per_page)
    stats("pdfplumber", pl_total, pl_per_page)


if __name__ == "__main__":
    main()
