
import paddle, paddleocr
paddle.utils.run_check()                # should print “…is installed successfully!” :contentReference[oaicite:1]{index=1}
print("Visible device:", paddle.device.get_device())  # expect gpu:0
ocr = paddleocr.PaddleOCR(use_angle_cls=False, lang="en")
print("PaddleOCR initialised OK")
