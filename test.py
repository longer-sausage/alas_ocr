import cv2
from rapidocr import RapidOCR, OCRVersion

cn = RapidOCR(
    params={
        "Global.use_det": False,
        "Global.use_cls": False,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.model_path": "./cn/cn.onnx",
        "Rec.rec_keys_path": "./cn/ppocrv5_dict.txt"
    }
)

num = RapidOCR(
    params={
        "Global.use_det": False,
        "Global.use_cls": False,
        "Rec.ocr_version": OCRVersion.PPOCRV4,
        "Rec.model_path": "./num/num.onnx",
        "Rec.rec_keys_path": "./num/en_dict.txt"
    }
)

img_path = "test.png"
img = cv2.imread(img_path)
result = num(img)

print("识别结果:", result.txts[0])
print("分数:", result.scores[0])
print("耗时:", result.elapse)