import os
import re

def remove_emojis_and_logs():
    app_path = '/Users/nguyenvan/AI_ML/CLASSIFICATION/src/gui/app.py'
    clf_path = '/Users/nguyenvan/AI_ML/CLASSIFICATION/src/inference/classifier.py'
    
    # Danh sách icon cần xoá
    emojis = [
        '🌿', '📋', '📷', '📸', '📹', '🔄', '🚪', '▶️', '⏹️', '💾', '📁', '📊', 
        '✅', '⚠️', '🤖', '⏸️', 'ℹ️', '📈', '🚀', '🎓', '⏱️', '📤', '🗑️', '📦', 
        '💡', '⚙️', '♻️', '📝', '📌', '🎯', '⭐', 'ℹ', '✔'
    ]
    
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    for emoji in emojis:
        content = content.replace(emoji + " ", "")
        content = content.replace(emoji, "")
        
    # Xoá các câu print cụ thể trong app.py
    content = re.sub(r'print\("✅ Model đã load!"\)', 'pass', content)
    content = re.sub(r'print\(f"📦 Load YOLO model.*?"\)', 'pass', content)
    content = re.sub(r'print\("✅ YOLO đã sẵn sàng!"\)', 'pass', content)
    
    with open(app_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    # Xử lý classifier.py
    with open(clf_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    content = re.sub(r'print\("✅ Đã load model CNN \(\.h5\) thành công!"\)', 'pass', content)
    content = re.sub(r'print\("✅ Đã load model SVM \(\.pkl\) thành công!"\)', 'pass', content)
    content = re.sub(r'print\(f"📂 Đang load model từ: {model_path}"\)', 'pass', content)
    
    with open(clf_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    remove_emojis_and_logs()
    print("Done")
