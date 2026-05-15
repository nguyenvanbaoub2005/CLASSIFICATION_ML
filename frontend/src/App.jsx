import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Leaf, ScanLine, BookOpen, Trophy } from 'lucide-react';
import SmartScanner from './components/SmartScanner';
import Encyclopedia from './components/Encyclopedia';
import Dashboard from './components/Dashboard';

function Navigation() {
  const location = useLocation();
  
  return (
    <nav className="navbar glass-panel">
      <Link to="/" className="nav-brand">
        <Leaf size={28} />
        EcoVision AI
      </Link>
      <div className="nav-links">
        <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>
          <ScanLine size={20} />
          Quét Rác
        </Link>
        <Link to="/dictionary" className={`nav-link ${location.pathname === '/dictionary' ? 'active' : ''}`}>
          <BookOpen size={20} />
          Từ Điển
        </Link>
        <Link to="/dashboard" className={`nav-link ${location.pathname === '/dashboard' ? 'active' : ''}`}>
          <Trophy size={20} />
          Thành Tích
        </Link>
      </div>
    </nav>
  );
}

function App() {
  return (
    <Router>
      <div className="app-container">
        <Navigation />
        <Routes>
          <Route path="/" element={<SmartScanner />} />
          <Route path="/dictionary" element={<Encyclopedia />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
