import { useState, useRef, useEffect } from 'react'
import { Send, Bot, Clock, Cloud, LayoutGrid, Battery, Calculator, Play, Search, Camera, Volume2, VolumeX, Lock, Trash2, Globe, Folder, Terminal, Monitor, MessageSquare, Info, Moon, Sun, Mic, MicOff, Copy, Download, Check } from 'lucide-react'

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

const QUICK_ACTIONS = [
  { label: 'Time', icon: <Clock size={16} />, query: 'what time is it' },
  { label: 'Weather', icon: <Cloud size={16} />, query: 'weather' },
  { label: 'System', icon: <Monitor size={16} />, query: 'how is my pc' },
  { label: 'Screenshot', icon: <Camera size={16} />, query: 'take a screenshot' },
  { label: 'Search', icon: <Search size={16} />, query: 'google search' },
  { label: 'YouTube', icon: <Play size={16} />, query: 'play music' },
  { label: 'Clear Chat', icon: <Trash2 size={16} />, query: 'CLEAR_CHAT' },
  { label: 'Calculator', icon: <Calculator size={16} />, query: 'open calculator' },
  { label: 'Lock PC', icon: <Lock size={16} />, query: 'lock my pc' },
  { label: 'Clean Bin', icon: <Trash2 size={16} />, query: 'empty recycle bin' },
  { label: 'Personalize', icon: <LayoutGrid size={16} />, query: 'PERSONALIZE_OPEN' },
];

interface UserDetails {
  name: string;
  gmail: string;
  linkedin: string;
  github: string;
  instagram: string;
  portfolio: string;
  other: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>(() => {
    const saved = localStorage.getItem('bugg_chat_history');
    if (saved) return JSON.parse(saved);
    
    const hour = new Date().getHours();
    const greeting = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
    const userDetailsSaved = localStorage.getItem('bugg_user_details');
    const name = userDetailsSaved ? JSON.parse(userDetailsSaved).name : "";
    return [{ 
      role: 'assistant', 
      content: `${greeting}${name ? ', ' + name : ''}! I'm Bugg AI, your advanced digital assistant. How can I help you today?`,
      timestamp: new Date().toLocaleTimeString()
    }];
  });
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showSplash, setShowSplash] = useState(true);
  const [isDarkMode, setIsDarkMode] = useState(() => localStorage.getItem('bugg_dark_mode') === 'true');
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [copiedId, setCopiedId] = useState<number | null>(null);
  
  const [showPersonalize, setShowPersonalize] = useState(false);
  const [userDetails, setUserDetails] = useState<UserDetails>(() => {
    const saved = localStorage.getItem('bugg_user_details');
    return saved ? JSON.parse(saved) : { name: '', gmail: '', linkedin: '', github: '', instagram: '', portfolio: '', other: '' };
  });
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    localStorage.setItem('bugg_chat_history', JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    localStorage.setItem('bugg_dark_mode', isDarkMode.toString());
  }, [isDarkMode]);

  useEffect(() => {
    localStorage.setItem('bugg_user_details', JSON.stringify(userDetails));
  }, [userDetails]);

  useEffect(() => {
    const timer = setTimeout(() => setShowSplash(false), 3000);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    if (!isLoading && !showSplash) {
      inputRef.current?.focus();
    }
  }, [messages, isLoading, showSplash]);

  // Voice Recognition Setup
  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setInput(transcript);
        setIsListening(false);
        handleSend(transcript);
      };

      recognitionRef.current.onerror = () => setIsListening(false);
      recognitionRef.current.onend = () => setIsListening(false);
    }
  }, []);

  const toggleListening = () => {
    if (isListening) {
      recognitionRef.current?.stop();
    } else {
      setIsListening(true);
      recognitionRef.current?.start();
    }
  };

  const speak = (text: string) => {
    if (!isVoiceEnabled) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.1;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
  };

  const handleSend = async (textOverride?: string) => {
    let userMessage = (textOverride || input).trim();
    if (!userMessage || isLoading) return;

    if (userMessage === 'PERSONALIZE_OPEN') {
      setShowPersonalize(true);
      return;
    }

    if (userMessage === 'CLEAR_CHAT') {
      clearHistory();
      return;
    }

    // Check for personalized commands
    const lowerMsg = userMessage.toLowerCase();
    if (lowerMsg.includes('open github') && userDetails.github) {
      window.open(userDetails.github, '_blank');
      setMessages(prev => [...prev, { role: 'user', content: userMessage, timestamp: new Date().toLocaleTimeString() }, { role: 'assistant', content: `Opening your GitHub profile, ${userDetails.name || 'sir'}.`, timestamp: new Date().toLocaleTimeString() }]);
      setInput('');
      return;
    } else if (lowerMsg.includes('open linkedin') && userDetails.linkedin) {
      window.open(userDetails.linkedin, '_blank');
      setMessages(prev => [...prev, { role: 'user', content: userMessage, timestamp: new Date().toLocaleTimeString() }, { role: 'assistant', content: `Opening your LinkedIn profile, ${userDetails.name || 'sir'}.`, timestamp: new Date().toLocaleTimeString() }]);
      setInput('');
      return;
    } else if (lowerMsg.includes('open gmail') && userDetails.gmail) {
      window.open(`https://mail.google.com/mail/?authuser=${userDetails.gmail}`, '_blank');
      setMessages(prev => [...prev, { role: 'user', content: userMessage, timestamp: new Date().toLocaleTimeString() }, { role: 'assistant', content: `Opening your Gmail, ${userDetails.name || 'sir'}.`, timestamp: new Date().toLocaleTimeString() }]);
      setInput('');
      return;
    } else if (lowerMsg.includes('open instagram') && userDetails.instagram) {
      window.open(userDetails.instagram, '_blank');
      setMessages(prev => [...prev, { role: 'user', content: userMessage, timestamp: new Date().toLocaleTimeString() }, { role: 'assistant', content: `Opening your Instagram, ${userDetails.name || 'sir'}.`, timestamp: new Date().toLocaleTimeString() }]);
      setInput('');
      return;
    } else if (lowerMsg.includes('open portfolio') && userDetails.portfolio) {
      window.open(userDetails.portfolio, '_blank');
      setMessages(prev => [...prev, { role: 'user', content: userMessage, timestamp: new Date().toLocaleTimeString() }, { role: 'assistant', content: `Opening your Portfolio, ${userDetails.name || 'sir'}.`, timestamp: new Date().toLocaleTimeString() }]);
      setInput('');
      return;
    }

    setInput('');
    const newMsg: Message = { role: 'user', content: userMessage, timestamp: new Date().toLocaleTimeString() };
    setMessages(prev => [...prev, newMsg]);
    setIsLoading(true);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8001';
      const response = await fetch(`${apiUrl}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: userMessage }),
      });
      const data = await response.json();
      
      if (data.response === 'WHAT_TO_SEARCH') {
        const assistantMsg: Message = { role: 'assistant', content: 'What would you like to search for?', timestamp: new Date().toLocaleTimeString() };
        setMessages(prev => [...prev, assistantMsg]);
        speak('What would you like to search for?');
      } else if (data.response === 'WHAT_TO_PLAY') {
        const assistantMsg: Message = { role: 'assistant', content: 'What song or video should I play on YouTube?', timestamp: new Date().toLocaleTimeString() };
        setMessages(prev => [...prev, assistantMsg]);
        speak('What song or video should I play on YouTube?');
      } else {
        const assistantMsg: Message = { role: 'assistant', content: data.response, timestamp: new Date().toLocaleTimeString() };
        setMessages(prev => [...prev, assistantMsg]);
        speak(data.response);
      }
    } catch (error) {
      const errorMsg = 'Connection failed. Please ensure the backend is running.';
      setMessages(prev => [...prev, { role: 'assistant', content: errorMsg, timestamp: new Date().toLocaleTimeString() }]);
      speak(errorMsg);
    } finally {
      setIsLoading(false);
    }
  };

  const clearHistory = () => {
    if (window.confirm('Clear all chat history?')) {
      setMessages([{ 
        role: 'assistant', 
        content: "History cleared. How can I help you now?",
        timestamp: new Date().toLocaleTimeString()
      }]);
    }
  };

  const copyToClipboard = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedId(index);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const downloadHistory = () => {
    const content = messages.map(m => `[${m.timestamp}] ${m.role.toUpperCase()}: ${m.content}`).join('\n\n');
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bugg_chat_history_${new Date().toISOString().slice(0,10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (showSplash) {
    return (
      <div className={`splash-screen ${isDarkMode ? 'dark' : ''}`}>
        <div className="netflix-zoom">
          <img src="/bugg_logo_orange.svg" alt="Bugg AI" className="splash-logo" />
        </div>
        <style>{`
          .splash-screen {
            position: fixed; top: 0; left: 0;
            height: 100vh; width: 100vw; background: #fff;
            display: flex; align-items: center; justify-content: center;
            overflow: hidden; z-index: 9999;
          }
          .splash-screen.dark { background: #0f172a; }
          .netflix-zoom {
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            animation: zoomOut 2.5s ease-in-out forwards;
          }
          .splash-logo { width: 500px; max-width: 80vw; }
          @keyframes zoomOut {
            0% { transform: scale(0.5); opacity: 0; }
            50% { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(1); opacity: 1; filter: blur(0px); }
          }
        `}</style>
      </div>
    );
  }

  return (
    <div className={`main-app ${isDarkMode ? 'dark-mode' : ''}`}>
      {/* Navigation Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <img src="/bugg_logo_orange.svg" alt="Logo" />
          <span>Bugg AI</span>
        </div>
        
        <div className="nav-section">
          <div className="section-header">
            <p className="section-title">Automation</p>
            <div className="sidebar-tools">
               <Download size={14} className="tool-btn" onClick={downloadHistory} title="Download History" />
               <Trash2 size={14} className="tool-btn danger" onClick={clearHistory} title="Clear History" />
            </div>
          </div>
          <div className="action-grid">
            {QUICK_ACTIONS.map((action, i) => (
              <button key={i} onClick={() => handleSend(action.query)} className="action-btn">
                {action.icon}
                <span>{action.label}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="system-status">
          <div className="status-item">
            <div className="dot online"></div>
            <span>Neural Link Active</span>
          </div>
          <p className="version">v3.5 Professional</p>
        </div>
      </aside>

      {/* Main Chat Interface */}
      <main className="chat-interface">
        <header className="chat-header">
          <div className="mobile-brand">
            <img src="/bugg_logo_orange.svg" alt="Logo" />
            <span>Bugg AI</span>
          </div>
          <div className="header-info">
            <h2>Digital Assistant</h2>
            <p>Ready to automate your digital world</p>
          </div>
          <div className="header-actions">
            <button className="icon-toggle" onClick={() => setIsVoiceEnabled(!isVoiceEnabled)} title={isVoiceEnabled ? "Mute Voice" : "Enable Voice"}>
              {isVoiceEnabled ? <Volume2 size={20} color="#ff6b00" /> : <VolumeX size={20} color="#888" />}
            </button>
            <button className="icon-toggle" onClick={() => setIsDarkMode(!isDarkMode)} title="Toggle Theme">
              {isDarkMode ? <Sun size={20} color="#ff6b00" /> : <Moon size={20} color="#888" />}
            </button>
            <div className="divider"></div>
            <Info size={20} color="#888" />
          </div>
        </header>

        <div className="chat-content">
          {messages.map((m, i) => (
            <div key={i} className={`message-wrapper ${m.role}`}>
              <div className="message-box">
                {m.content}
                <div className="message-meta">
                  <span className="timestamp">{m.timestamp}</span>
                  <button className="copy-msg" onClick={() => copyToClipboard(m.content, i)}>
                    {copiedId === i ? <Check size={12} /> : <Copy size={12} />}
                  </button>
                </div>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="message-wrapper assistant">
              <div className="message-box typing">
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="dot"></span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <footer className="chat-footer">
          <div className="input-bar">
            <button onClick={toggleListening} className={`mic-btn ${isListening ? 'listening' : ''}`}>
              {isListening ? <MicOff size={20} /> : <Mic size={20} />}
            </button>
            <input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder={isListening ? "Listening..." : "Command your assistant..."}
              disabled={isLoading}
            />
            <button onClick={() => handleSend()} disabled={isLoading || !input.trim()} className="send-btn">
              <Send size={20} />
            </button>
          </div>
        </footer>
      </main>

      {/* Personalize Modal */}
      {showPersonalize && (
        <div className="modal-overlay">
          <div className={`modal-content personalize-modal ${isDarkMode ? 'dark' : ''}`}>
            <div className="modal-header">
              <h3>Personalize Bugg AI</h3>
              <button className="close-modal" onClick={() => setShowPersonalize(false)}>×</button>
            </div>
            
            <div className="modal-body">
              <div className="input-grid">
                <div className="input-group full">
                  <label><Terminal size={14} /> Your Name</label>
                  <input 
                    type="text" 
                    value={userDetails.name} 
                    onChange={(e) => setUserDetails({...userDetails, name: e.target.value})}
                    placeholder="Enter your name"
                  />
                </div>
                <div className="input-group">
                  <label><Globe size={14} /> Gmail</label>
                  <input 
                    type="text" 
                    value={userDetails.gmail} 
                    onChange={(e) => setUserDetails({...userDetails, gmail: e.target.value})}
                    placeholder="yourname@gmail.com"
                  />
                </div>
                <div className="input-group">
                  <label><Monitor size={14} /> Portfolio</label>
                  <input 
                    type="text" 
                    value={userDetails.portfolio} 
                    onChange={(e) => setUserDetails({...userDetails, portfolio: e.target.value})}
                    placeholder="https://yourportfolio.com"
                  />
                </div>
                <div className="input-group">
                  <label><Info size={14} /> LinkedIn</label>
                  <input 
                    type="text" 
                    value={userDetails.linkedin} 
                    onChange={(e) => setUserDetails({...userDetails, linkedin: e.target.value})}
                    placeholder="LinkedIn URL"
                  />
                </div>
                <div className="input-group">
                  <label><Terminal size={14} /> GitHub</label>
                  <input 
                    type="text" 
                    value={userDetails.github} 
                    onChange={(e) => setUserDetails({...userDetails, github: e.target.value})}
                    placeholder="GitHub URL"
                  />
                </div>
                <div className="input-group">
                  <label><Camera size={14} /> Instagram</label>
                  <input 
                    type="text" 
                    value={userDetails.instagram} 
                    onChange={(e) => setUserDetails({...userDetails, instagram: e.target.value})}
                    placeholder="Instagram URL"
                  />
                </div>
                <div className="input-group">
                  <label><MessageSquare size={14} /> Other Info</label>
                  <input 
                    type="text" 
                    value={userDetails.other} 
                    onChange={(e) => setUserDetails({...userDetails, other: e.target.value})}
                    placeholder="Any other details..."
                  />
                </div>
              </div>
            </div>

            <div className="modal-actions">
              <button className="reset-btn" onClick={() => setUserDetails({ name: '', gmail: '', linkedin: '', github: '', instagram: '', portfolio: '', other: '' })}>Reset All</button>
              <button className="save-btn" onClick={() => setShowPersonalize(false)}>Save Preferences</button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        :root {
          --primary: #ff6b00;
          --bg-main: #ffffff;
          --sidebar-bg: #ffffff;
          --chat-bg: #ffffff;
          --header-bg: #ffffff;
          --text-main: #1a1a1a;
          --text-sub: #6c757d;
          --border: #eef0f2;
          --msg-user: var(--primary);
          --msg-assistant: #f1f3f5;
          --input-bg: #f8f9fa;
        }

        .dark-mode {
          --bg-main: #0f172a;
          --sidebar-bg: #1e293b;
          --chat-bg: #0f172a;
          --header-bg: #1e293b;
          --text-main: #f8fafc;
          --text-sub: #94a3b8;
          --border: #334155;
          --msg-user: #ff6b00;
          --msg-assistant: #334155;
          --input-bg: #1e293b;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: var(--bg-main); transition: background 0.3s; }

        .main-app {
          display: flex; height: 100vh; width: 100vw; overflow: hidden; background: var(--bg-main); color: var(--text-main);
        }

        /* Sidebar Styling */
        .sidebar {
          width: 320px; background: var(--sidebar-bg); border-right: 1px solid var(--border);
          display: flex; flex-direction: column; padding: 25px;
          transition: all 0.3s ease;
        }

        .brand {
          display: flex; align-items: center; gap: 18px; margin-bottom: 40px;
          flex-wrap: nowrap;
        }
        .brand img { width: 100px; height: auto; flex-shrink: 0; }
        .brand span { 
          font-size: 2.6rem; 
          font-weight: 900; 
          color: var(--primary); 
          letter-spacing: -1.5px; 
          white-space: nowrap;
        }

        .section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .sidebar-tools { display: flex; gap: 10px; }
        .tool-btn { cursor: pointer; color: var(--text-sub); transition: color 0.2s; }
        .tool-btn:hover { color: var(--primary); }
        .tool-btn.danger:hover { color: #ef4444; }

        .section-title {
          font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px;
          color: var(--text-sub); font-weight: 700;
        }

        .action-grid {
          display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
        }

        .action-btn {
          display: flex; flex-direction: column; align-items: center; justify-content: center;
          gap: 8px; padding: 15px 10px; border: 1px solid var(--border); border-radius: 12px;
          background: transparent; color: var(--text-main); cursor: pointer; transition: all 0.2s;
          font-size: 0.8rem; font-weight: 600;
        }
        .action-btn:hover {
          border-color: var(--primary); color: var(--primary); background: rgba(255,107,0,0.05);
          transform: translateY(-2px); box-shadow: 0 4px 12px rgba(255,107,0,0.1);
        }

        .system-status { margin-top: auto; padding-top: 20px; border-top: 1px solid var(--border); }
        .status-item { display: flex; align-items: center; gap: 8px; font-size: 0.85rem; font-weight: 600; color: #22c55e; }
        .dot { width: 8px; height: 8px; border-radius: 50%; }
        .dot.online { background: #22c55e; box-shadow: 0 0 8px #22c55e; }
        .version { font-size: 0.7rem; color: var(--text-sub); margin-top: 5px; }

        /* Chat Interface */
        .chat-interface {
          flex: 1; display: flex; flex-direction: column; background: var(--chat-bg);
          width: 100%;
        }

        .chat-header {
          padding: 20px 40px; border-bottom: 1px solid var(--border); background: var(--header-bg);
          display: flex; justify-content: space-between; align-items: center;
        }
        .mobile-brand { display: none; align-items: center; gap: 10px; }
        .mobile-brand img { width: 55px; height: auto; }
        .mobile-brand span { font-size: 1.5rem; font-weight: 900; color: var(--primary); }

        .chat-header h2 { font-size: 1.25rem; font-weight: 700; color: var(--text-main); }
        .chat-header p { font-size: 0.85rem; color: var(--text-sub); }
        
        .header-actions { display: flex; align-items: center; gap: 15px; }
        .icon-toggle { background: none; border: none; cursor: pointer; display: flex; align-items: center; justify-content: center; padding: 5px; border-radius: 8px; transition: background 0.2s; }
        .icon-toggle:hover { background: rgba(0,0,0,0.05); }
        .dark-mode .icon-toggle:hover { background: rgba(255,255,255,0.05); }
        .divider { width: 1px; height: 24px; background: var(--border); margin: 0 5px; }

        .chat-content {
          flex: 1; overflow-y: auto; padding: 40px; display: flex; flex-direction: column; gap: 25px;
        }

        .message-wrapper { display: flex; width: 100%; }
        .message-wrapper.user { justify-content: flex-end; }
        .message-wrapper.assistant { justify-content: flex-start; }

        .message-box {
          max-width: 75%; padding: 15px 22px; border-radius: 18px;
          font-size: 0.95rem; line-height: 1.6; position: relative;
          box-shadow: 0 2px 10px rgba(0,0,0,0.02);
          animation: slideUp 0.3s ease-out;
          display: flex; flex-direction: column;
        }
        .user .message-box { background: var(--msg-user); color: #fff; border-bottom-right-radius: 4px; }
        .assistant .message-box { background: var(--msg-assistant); color: var(--text-main); border-bottom-left-radius: 4px; }

        .message-meta { display: flex; align-items: center; justify-content: space-between; margin-top: 8px; opacity: 0.7; font-size: 0.7rem; }
        .copy-msg { background: none; border: none; color: inherit; cursor: pointer; padding: 2px; border-radius: 4px; }
        .copy-msg:hover { background: rgba(0,0,0,0.1); }

        .typing { display: flex; gap: 5px; padding: 15px 25px; }
        .typing .dot { background: var(--primary); animation: bounce 1.4s infinite ease-in-out; }
        .typing .dot:nth-child(1) { animation-delay: -0.32s; }
        .typing .dot:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
        @keyframes slideUp { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }

        .chat-footer { padding: 30px 40px; border-top: 1px solid var(--border); }
        .input-bar {
          max-width: 1000px; margin: 0 auto; display: flex; gap: 15px; align-items: center;
          background: var(--input-bg); padding: 8px; border-radius: 16px; border: 1px solid var(--border);
        }
        .input-bar input {
          flex: 1; border: none; background: transparent; padding: 12px 20px;
          font-size: 1rem; outline: none; color: var(--text-main);
        }
        .send-btn, .mic-btn {
          width: 48px; height: 48px; border-radius: 12px; border: none;
          background: var(--primary); color: #fff; cursor: pointer;
          display: flex; align-items: center; justify-content: center;
          transition: transform 0.2s, box-shadow 0.2s;
        }
        .mic-btn { background: #64748b; }
        .mic-btn.listening { background: #ef4444; animation: pulse 1.5s infinite; }
        .send-btn:hover, .mic-btn:hover { transform: scale(1.05); }
        .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }

        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); } 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); } }

        /* Modal Styles */
        .modal-overlay {
          position: fixed; top: 0; left: 0; width: 100%; height: 100%;
          background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center;
          z-index: 1000; backdrop-filter: blur(8px);
        }
        .personalize-modal {
          width: 90%; max-width: 650px; padding: 0; overflow: hidden;
          background: #fff; border-radius: 24px;
        }
        .personalize-modal.dark { background: #1e293b; color: #f8fafc; }
        
        .modal-header {
          padding: 25px 30px; border-bottom: 1px solid var(--border);
          display: flex; justify-content: space-between; align-items: center;
          background: rgba(255,107,0,0.05);
        }
        .modal-header h3 { font-size: 1.5rem; font-weight: 800; color: var(--primary); }
        .close-modal { background: none; border: none; font-size: 2rem; cursor: pointer; color: var(--text-sub); }

        .modal-body { padding: 30px; max-height: 70vh; overflow-y: auto; }
        
        .input-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .input-group.full { grid-column: span 2; }
        
        .input-group label {
          display: flex; align-items: center; gap: 8px; margin-bottom: 8px;
          font-size: 0.85rem; font-weight: 700; color: var(--text-sub);
        }
        .input-group input {
          width: 100%; padding: 12px 16px; border-radius: 12px; border: 1.5px solid var(--border);
          background: var(--input-bg); color: var(--text-main); outline: none;
          transition: border-color 0.2s;
        }
        .input-group input:focus { border-color: var(--primary); }

        .modal-actions {
          padding: 20px 30px; border-top: 1px solid var(--border);
          display: flex; justify-content: space-between; align-items: center;
          background: var(--input-bg);
        }
        
        .reset-btn { background: none; border: none; color: #ef4444; font-weight: 600; cursor: pointer; }
        .save-btn {
          background: var(--primary); color: #fff; border: none; padding: 12px 30px;
          border-radius: 12px; font-weight: 700; cursor: pointer;
          box-shadow: 0 4px 12px rgba(255,107,0,0.2);
        }

        @keyframes scaleUp { from { transform: scale(0.95); opacity: 0; } to { transform: scale(1); opacity: 1; } }

        /* Responsive Design */
        @media (max-width: 1024px) {
          .sidebar { width: 260px; }
          .brand span { font-size: 1.8rem; }
          .brand img { width: 60px; }
        }

        @media (max-width: 850px) {
          .sidebar { width: 80px; padding: 20px 10px; }
          .brand span, .section-title, .action-grid span, .system-status, .sidebar-tools { display: none; }
          .action-grid { grid-template-columns: 1fr; }
          .brand { justify-content: center; margin-bottom: 30px; }
          .brand img { width: 60px; }
          .action-btn { padding: 15px; }
        }

        @media (max-width: 600px) {
          .sidebar { display: none; }
          .mobile-brand { display: flex; }
          .header-info { display: none; }
          .chat-header { padding: 15px 20px; }
          .chat-content { padding: 20px; gap: 15px; }
          .chat-footer { padding: 15px 20px; }
          .message-box { max-width: 90%; padding: 12px 16px; }
          .input-bar { gap: 8px; padding: 5px; }
          .input-bar input { padding: 8px 12px; font-size: 0.9rem; }
          .send-btn, .mic-btn { width: 40px; height: 40px; }
          .input-grid { grid-template-columns: 1fr; }
          .input-group.full { grid-column: span 1; }
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #e0e0e0; border-radius: 10px; }
        .dark-mode ::-webkit-scrollbar-thumb { background: #334155; }
      `}</style>
    </div>
  )
}

export default App
