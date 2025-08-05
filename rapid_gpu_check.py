
#!/usr/bin/env python
"""
Sanity-check that RapidOCR is using onnxruntime-gpu.

Requires:
    pip install rapidocr-onnxruntime onnxruntime-gpu opencv-python-headless
"""

import time
import numpy as np
import onnxruntime as ort
from rapidocr_onnxruntime import RapidOCR

print("ORT providers :", ort.get_available_providers())
assert "CUDAExecutionProvider" in ort.get_available_providers(), (
    "onnxruntime-gpu is not active; reinstall with: pip install onnxruntime-gpu"
)

# build OCR engine; providers list forces CUDA first, CPU fallback second
t0 = time.time()
ocr = RapidOCR(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print(f"OCR initialised in {time.time()-t0:.2f}s")

# create a blank white image
blank = np.full((640, 640, 3), 255, dtype=np.uint8)

# two runs to show warm-up vs steady state
for i in range(2):
    t0 = time.time()
    _, _ = ocr(blank)           # returns (result, elapsed)
    print(f"run {i+1}: {(time.time()-t0)*1000:.1f} ms")
