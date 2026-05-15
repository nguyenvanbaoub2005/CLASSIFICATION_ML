import { motion } from 'framer-motion';

const wasteData = [
  { id: 'plastic', name: 'Nhựa', icon: '🥤', color: '#3b82f6', desc: 'Mất tới 500 năm để phân hủy. Tái chế nhựa giúp tiết kiệm 80% năng lượng so với làm nhựa mới.' },
  { id: 'paper', name: 'Giấy', icon: '📄', color: '#eab308', desc: 'Tái chế 1 tấn giấy cứu được 17 cây xanh và 26,500 lít nước sạch.' },
  { id: 'glass', name: 'Thủy tinh', icon: '🍾', color: '#10b981', desc: 'Thủy tinh có thể tái chế 100% không giới hạn số lần mà không giảm chất lượng.' },
  { id: 'metal', name: 'Kim loại', icon: '🥫', color: '#64748b', desc: 'Tái chế nhôm tiết kiệm 95% năng lượng so với việc khai thác quặng bauxite mới.' },
  { id: 'organic', name: 'Hữu cơ', icon: '🍃', color: '#15803d', desc: 'Rác hữu cơ khi ủ thành phân compost giúp đất tơi xốp và giảm phát thải khí methane từ bãi rác.' },
  { id: 'cardboard', name: 'Bìa carton', icon: '📦', color: '#d97706', desc: 'Carton rất dễ tái chế thành các hộp mới, giấy gói bảo vệ môi trường.' },
  { id: 'trash', name: 'Rác chung', icon: '🗑️', color: '#ef4444', desc: 'Các loại rác không thể tái chế. Cần hạn chế tạo ra rác loại này.' },
];

export default function Encyclopedia() {
  return (
    <div>
      <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
        <h1 style={{ color: 'var(--primary)', marginBottom: '1rem' }}>Từ Điển Rác Thải</h1>
        <p style={{ color: 'var(--text-muted)', maxWidth: '600px', margin: '0 auto' }}>
          Hiểu rõ về từng loại rác là bước đầu tiên để bảo vệ môi trường. Khám phá thời gian phân hủy và giá trị tái chế của chúng.
        </p>
      </div>

      <div className="grid-cards">
        {wasteData.map((item, index) => (
          <motion.div 
            key={item.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="glass-panel eco-card"
            style={{ borderTop: `4px solid ${item.color}` }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
              <span style={{ fontSize: '2.5rem' }}>{item.icon}</span>
              <h3 style={{ fontSize: '1.25rem', color: 'var(--text-main)' }}>{item.name}</h3>
            </div>
            <p style={{ color: 'var(--text-muted)', lineHeight: 1.6 }}>{item.desc}</p>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
