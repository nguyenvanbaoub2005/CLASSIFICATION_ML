import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Doughnut } from 'react-chartjs-2';
import { Trophy, Star, Target, Zap } from 'lucide-react';

ChartJS.register(ArcElement, Tooltip, Legend);

export default function Dashboard() {
  const [stats, setStats] = useState({ points: 0, scans: 0, history: [] });

  useEffect(() => {
    const savedStats = JSON.parse(localStorage.getItem('ecoStats'));
    if (savedStats) setStats(savedStats);
  }, []);

  const level = stats.points < 50 ? 'Mầm Non' : stats.points < 200 ? 'Hiệp Sĩ' : 'Đại Sứ Môi Trường';

  const chartData = {
    labels: ['Nhựa', 'Giấy', 'Thủy tinh', 'Kim loại', 'Bìa cứng', 'Hữu cơ', 'Rác chung'],
    datasets: [
      {
        data: [
          stats.classCounts?.plastic || 0,
          stats.classCounts?.paper || 0,
          stats.classCounts?.glass || 0,
          stats.classCounts?.metal || 0,
          stats.classCounts?.cardboard || 0,
          stats.classCounts?.organic || 0,
          stats.classCounts?.trash || 0,
        ],
        backgroundColor: [
          '#3b82f6', // plastic
          '#eab308', // paper
          '#10b981', // glass
          '#64748b', // metal
          '#d97706', // cardboard
          '#15803d', // organic
          '#ef4444', // trash
        ],
        borderWidth: 0,
      },
    ],
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
      
      {/* Gamification Stats */}
      <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="glass-panel" style={{ padding: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '2rem' }}>
          <Trophy size={32} color="#f59e0b" />
          <div>
            <h2 style={{ margin: 0 }}>Hồ sơ Sinh thái</h2>
            <p style={{ color: 'var(--text-muted)', margin: 0 }}>Cấp độ: <strong style={{ color: 'var(--primary)' }}>{level}</strong></p>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '2rem' }}>
          <div style={{ background: 'rgba(59, 130, 246, 0.1)', padding: '1.5rem', borderRadius: '1rem', textAlign: 'center' }}>
            <Star color="#3b82f6" style={{ margin: '0 auto 0.5rem' }} />
            <h3 style={{ fontSize: '2rem', color: '#3b82f6', margin: 0 }}>{stats.points}</h3>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', margin: 0 }}>Eco Points</p>
          </div>
          <div style={{ background: 'rgba(16, 185, 129, 0.1)', padding: '1.5rem', borderRadius: '1rem', textAlign: 'center' }}>
            <Target color="#10b981" style={{ margin: '0 auto 0.5rem' }} />
            <h3 style={{ fontSize: '2rem', color: '#10b981', margin: 0 }}>{stats.scans}</h3>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', margin: 0 }}>Lần quét đúng</p>
          </div>
        </div>

        <div>
          <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Zap size={20} color="#f59e0b" /> Lịch sử gần đây
          </h3>
          {stats.history.length === 0 ? (
            <p style={{ color: 'var(--text-muted)' }}>Bạn chưa quét rác lần nào.</p>
          ) : (
            <ul style={{ listStyle: 'none', padding: 0 }}>
              {stats.history.map((h, i) => (
                <li key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem 0', borderBottom: '1px solid var(--border)' }}>
                  <span style={{ fontWeight: 500 }}>{h.type}</span>
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>{h.date}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </motion.div>

      {/* Chart */}
      <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.1 }} className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <h2 style={{ width: '100%', marginBottom: '2rem' }}>Thống kê Phân loại</h2>
        <div style={{ width: '80%', maxWidth: '300px' }}>
          <Doughnut data={chartData} options={{ cutout: '70%' }} />
        </div>
        <p style={{ textAlign: 'center', marginTop: '2rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
          Phân loại rác thường xuyên giúp hệ thống AI thông minh hơn và lan tỏa ý thức xanh đến mọi người!
        </p>
      </motion.div>

    </div>
  );
}
