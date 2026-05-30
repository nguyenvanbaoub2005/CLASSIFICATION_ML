"""
predict.py — Đặt vào: scripts/predict.py
Dự đoán ảnh thực tế từ internet với GrabCut foreground mask
Yêu cầu: feature_utils.py phải nằm cùng thư mục scripts/
"""

import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

from feature_utils import extract_features   # cùng thư mục scripts/


def predict_image(model_path: str, image_input):
    """
    Dự đoán 1 ảnh với GrabCut mask (tốt hơn cho ảnh internet).
    image_input: đường dẫn file (str) hoặc numpy array RGB
    Trả về: (class_name, confidence, proba_dict)
    """
    bundle    = joblib.load(model_path)
    model     = bundle['model']
    scaler    = bundle['scaler']
    img_size  = bundle['img_size']
    classes   = bundle['classes']
    threshold = bundle.get('confidence_threshold', 0.50)

    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Không đọc được: {image_input}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image_input.copy()

    img = cv2.resize(img, img_size)

    # GrabCut — loại nền phức tạp khi predict
    feat = extract_features(img, use_grabcut=True).reshape(1, -1)
    feat_sc = scaler.transform(feat)

    proba      = model.predict_proba(feat_sc)[0]
    pred_idx   = proba.argmax()
    confidence = proba[pred_idx]
    pred_class = classes[pred_idx]
    proba_dict = {cls: float(p) for cls, p in zip(classes, proba)}

    if confidence < threshold:
        pred_class = f"uncertain ({pred_class}?)"

    return pred_class, float(confidence), proba_dict


def visualize_prediction(image_path: str, model_path: str):
    pred_class, confidence, proba_dict = predict_image(model_path, image_path)
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img)
    color = 'green' if 'uncertain' not in pred_class else 'orange'
    axes[0].set_title(f"Dự đoán: {pred_class}\nConfidence: {confidence*100:.1f}%",
                      color=color, fontsize=13, fontweight='bold')
    axes[0].axis('off')

    classes_sorted = sorted(proba_dict, key=proba_dict.get, reverse=True)
    values = [proba_dict[c] * 100 for c in classes_sorted]
    bar_colors = ['#F44336' if i == 0 else '#90CAF9' for i in range(len(classes_sorted))]
    axes[1].barh(classes_sorted[::-1], values[::-1], color=bar_colors[::-1])
    axes[1].set_xlabel('Confidence (%)')
    axes[1].set_title('Phân bố xác suất')
    axes[1].set_xlim(0, 105)
    for i, v in enumerate(values[::-1]):
        axes[1].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    out_path = os.path.splitext(image_path)[0] + '_prediction.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"\n🎯 {pred_class}  (confidence: {confidence*100:.1f}%)")
    print(f"🖼️  Kết quả → {out_path}")
    plt.show()


def batch_predict(model_path: str, image_dir: str):
    exts  = ('.png', '.jpg', '.jpeg', '.webp')
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(exts)]
    print(f"\n📂 {len(files)} ảnh trong {image_dir}\n")
    print(f"{'File':35s} {'Prediction':22s} {'Conf':>8s}")
    print("-" * 70)
    uncertain_count = 0
    for fname in sorted(files):
        try:
            pred, conf, _ = predict_image(model_path,
                                          os.path.join(image_dir, fname))
            flag = "⚠️ " if 'uncertain' in pred else "✅"
            print(f"{fname:35s} {pred:22s} {conf*100:>6.1f}%  {flag}")
            if 'uncertain' in pred:
                uncertain_count += 1
        except Exception as e:
            print(f"{fname:35s} ERROR: {e}")
    print(f"\n📊 {len(files)} ảnh | "
          f"{uncertain_count} uncertain ({uncertain_count/max(len(files),1)*100:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  required=True, help='Đường dẫn file .pkl')
    parser.add_argument('--image',  default=None,  help='1 ảnh đơn')
    parser.add_argument('--dir',    default=None,  help='Thư mục ảnh (batch)')
    args = parser.parse_args()

    if args.image:
        visualize_prediction(args.image, args.model)
    elif args.dir:
        batch_predict(args.model, args.dir)
    else:
        print("Dùng:  --image <path>  hoặc  --dir <folder>")
