import time, numpy as np, paddle
from paddleocr import PaddleOCR

paddle.utils.run_check()
print("Visible device :", paddle.device.get_device())

ocr = PaddleOCR(use_textline_orientation=False, lang="en")
print("OCR initialised")

blank = np.full((640, 640, 3), 255, dtype=np.uint8)
t0 = time.time()
_ = ocr.ocr(blank)          # <-- corrected call
print(f"Forward-pass time : {(time.time()-t0)*1000:.1f} ms")
