import { useState, useRef } from 'react';
import { UploadCloud, Camera, CheckCircle, Info, RefreshCw, ScanLine } from 'lucide-react';
import axios from 'axios';
import { motion } from 'framer-motion';

export default function SmartScanner() {
  const [imageSrc, setImageSrc] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [useCamera, setUseCamera] = useState(false);
  
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  // === HANDLE UPLOAD ===
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) processFile(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  };

  const processFile = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      setImageSrc(e.target.result);
      uploadImage(file);
    };
    reader.readAsDataURL(file);
  };

  // === HANDLE CAMERA ===
  const startCamera = async () => {
    setUseCamera(true);
    setImageSrc(null);
    setResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
      videoRef.current.srcObject = stream;
      streamRef.current = stream;
    } catch (err) {
      alert("Không thể mở camera. Vui lòng cấp quyền!");
      setUseCamera(false);
    }
  };

  const capturePhoto = () => {
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext("2d").drawImage(videoRef.current, 0, 0);
    const dataUrl = canvas.toDataURL("image/jpeg");
    setImageSrc(dataUrl);
    
    // Convert data url to file
    canvas.toBlob((blob) => {
      const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
      uploadImage(file);
    }, "image/jpeg");
    
    stopCamera();
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    setUseCamera(false);
  };

  // === API CALL ===
  const uploadImage = async (file) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const res = await axios.post(`${apiUrl}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(res.data.prediction);
      
      // Update local storage for dashboard gamification
      const stats = JSON.parse(localStorage.getItem('ecoStats')) || { points: 0, scans: 0, history: [], classCounts: {} };
      stats.scans += 1;
      
      // Give points if confident
      const predClass = res.data.prediction.class;
      if (['plastic', 'glass', 'metal', 'paper', 'cardboard'].includes(predClass)) {
        stats.points += 10;
      } else if (predClass === 'organic') {
        stats.points += 5;
      }
      
      // Update class counts for chart
      if (!stats.classCounts) stats.classCounts = {};
      stats.classCounts[predClass] = (stats.classCounts[predClass] || 0) + 1;
      
      // Save history limit 10
      stats.history.unshift({ date: new Date().toLocaleDateString(), type: res.data.prediction.class_name_vi });
      if(stats.history.length > 10) stats.history.pop();
      
      localStorage.setItem('ecoStats', JSON.stringify(stats));

    } catch (err) {
      console.error(err);
      alert("Lỗi khi kết nối tới AI. Hãy chắc chắn backend đang chạy!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="scanner-container">
      {/* Left Panel: Input Area */}
      <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="glass-panel" style={{ padding: '2rem' }}>
        <h2 style={{ marginBottom: '1.5rem', color: 'var(--primary)' }}>Trạm Quét Rác AI</h2>
        
        {!useCamera && !imageSrc && (
          <div 
            className="upload-zone"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current.click()}
          >
            <UploadCloud size={48} color="var(--primary)" style={{ marginBottom: '1rem' }} />
            <h3>Kéo thả ảnh rác vào đây</h3>
            <p style={{ color: 'var(--text-muted)', margin: '0.5rem 0' }}>hoặc bấm để chọn file từ máy</p>
            <input 
              type="file" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              accept="image/*"
              onChange={handleFileChange}
            />
          </div>
        )}

        {useCamera && (
          <div className="video-wrapper">
            <video ref={videoRef} autoPlay playsInline />
            <div style={{ position: 'absolute', bottom: '20px', display: 'flex', gap: '10px' }}>
              <button className="btn-primary" onClick={capturePhoto}>📸 Chụp</button>
              <button className="btn-outline" style={{ background: 'white' }} onClick={stopCamera}>Hủy</button>
            </div>
          </div>
        )}

        {imageSrc && !useCamera && (
          <div style={{ textAlign: 'center' }}>
            <img src={imageSrc} alt="Preview" className="preview-image" />
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', marginTop: '2rem' }}>
          {(!useCamera && !imageSrc) && (
            <button className="btn-primary" onClick={startCamera}>
              <Camera size={20} /> Mở Camera
            </button>
          )}
          {imageSrc && (
            <button className="btn-outline" onClick={() => { setImageSrc(null); setResult(null); }}>
              <RefreshCw size={20} /> Quét ảnh khác
            </button>
          )}
        </div>
      </motion.div>

      {/* Right Panel: Results */}
      <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="glass-panel result-card">
        {loading ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '1rem' }}>
            <RefreshCw size={40} color="var(--primary)" className="spin" style={{ animation: 'spin 1s linear infinite' }} />
            <h3>AI đang phân tích ảnh...</h3>
            <style>{`@keyframes spin { 100% { transform: rotate(360deg); } }`}</style>
          </div>
        ) : result ? (
          <>
            <div className="result-header">
              <span style={{ fontSize: '3rem' }}>{result.class === 'plastic' ? '🥤' : result.class === 'paper' ? '📄' : result.class === 'glass' ? '🍾' : result.class === 'metal' ? '🥫' : result.class === 'organic' ? '🍃' : result.class === 'cardboard' ? '📦' : '🗑️'}</span>
              <h2 style={{ fontSize: '2rem', margin: '0.5rem 0', color: result.is_confident ? 'var(--primary)' : '#f59e0b' }}>
                {result.class_name_vi.toUpperCase()}
              </h2>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', fontWeight: 600 }}>
                  <span>Độ tin cậy</span>
                  <span>{result.confidence.toFixed(1)}%</span>
                </div>
                <div className="confidence-bar-bg">
                  <div className="confidence-bar-fill" style={{ width: `${result.confidence}%`, backgroundColor: result.is_confident ? 'var(--primary)' : '#f59e0b' }}></div>
                </div>
              </div>
            </div>

            <div className="info-item">
              <CheckCircle className="info-icon" />
              <div>
                <h4 style={{ marginBottom: '0.25rem' }}>Cách xử lý</h4>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: 1.5 }}>
                  {result.class === 'plastic' ? 'Rửa sạch cặn bẩn, phơi khô và bỏ vào thùng rác tái chế màu xanh lá.' : 
                   result.class === 'organic' ? 'Sử dụng để ủ làm phân bón hữu cơ (compost) cho cây trồng.' :
                   result.class === 'glass' ? 'Bọc cẩn thận nếu bị vỡ. Bỏ vào đúng thùng thủy tinh tái chế.' :
                   'Phân loại đúng nơi quy định để giảm tải ô nhiễm.'}
                </p>
              </div>
            </div>

            <div className="info-item">
              <Info className="info-icon" />
              <div>
                <h4 style={{ marginBottom: '0.25rem' }}>Bạn có biết?</h4>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: 1.5 }}>
                  Bằng cách quét rác này, bạn vừa kiếm được điểm EcoPoint. Tái chế 1 tấn {result.class_name_vi.toLowerCase()} giúp cứu được rất nhiều tài nguyên thiên nhiên!
                </p>
              </div>
            </div>
          </>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)', textAlign: 'center' }}>
            <ScanLine size={64} style={{ opacity: 0.2, marginBottom: '1rem' }} />
            <h3 style={{ fontWeight: 500 }}>Chưa có kết quả</h3>
            <p>Hãy tải ảnh lên hoặc chụp bằng camera để AI phân tích nhé</p>
          </div>
        )}
      </motion.div>
    </div>
  );
}
