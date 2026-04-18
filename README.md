# 🤖 Bugg AI – Advanced Digital Assistant (v3.5 Pro)

**Bugg AI** is a professional-grade, full-stack digital assistant featuring a high-performance React frontend and a robust Python backend. It uses a hybrid intelligence architecture: a custom-trained neural network for local intent classification, an offline LLM (Ollama) for private queries, and Google Gemini for high-level online reasoning.

---

## 🚀 Key Features

### 💻 Modern Interface
- **React + Vite Dashboard**: Sleek, high-performance UI with **Dark/Light mode** support.
- **Voice-First Design**: Integrated Speech-to-Text and Text-to-Speech directly in the browser.
- **Quick Action Sidebar**: Instant access to common tasks with dynamic user prompts.
- **Personalize Hub**: Store your Gmail, LinkedIn, and GitHub profiles to enable one-click navigation and personalized commands.

### 🧠 Triple-Layer Intelligence
1. **Local Intent Engine**: High-speed classification using a trained Keras model for system tasks.
2. **Offline LLM (Ollama)**: Local Llama3 fallback for secure, private, and offline conversational AI.
3. **Cloud Intelligence (Gemini)**: State-of-the-art fallback for complex, real-time online queries.

### 🛠️ Automation & Tasks
- **System Control**: Mute/Unmute audio, take screenshots, and fetch real-time PC status (CPU/RAM/Battery).
- **Web Navigation**: Personalized "Open my GitHub/LinkedIn/Gmail" commands.
- **Music & Media**: Dynamic YouTube playback and Google searches.
- **Information**: Real-time weather, Wikipedia summaries, news headlines, and IP tracking.
- **Utilities**: To-do list management, alarms, jokes, and application launching.

---

## 📁 Project Structure

```text
AI-Virtual-Assistent-main/
├── backend/                # Python FastAPI Backend
│   ├── assistant_engine.py # Core logic & AI fallback layers
│   ├── api.py              # FastAPI server (Port 8001)
│   ├── train.py            # Model training script
│   ├── intents.json        # Intent training data
│   ├── chatbot.h5          # Trained Neural Net
│   └── bot.py              # Standalone voice-only bot
├── frontend/               # React + Vite + TypeScript
│   ├── src/                # App.tsx (Main UI Logic)
│   └── public/             # Assets & SVGs
├── scripts/                # Launch scripts (.bat / .sh)
├── .env                    # API Keys (Gemini, Weather)
└── README.md               # You are here
```

---

## ⚙️ Quick Start

### 1. Setup Backend
```bash
cd backend
pip install -r requirements.txt
python train.py      # Generate the model
python api.py        # Start the API server
```

### 2. Setup Frontend
```bash
cd frontend
npm install
npm run dev          # Start the dashboard
```

### 3. Configure `.env`
Create a `.env` in the root (or `backend/`) directory:
```env
GEMINI_API_KEY=your_key
OPENWEATHER_API_KEY=your_key
```

### 4. Optional: Offline AI
To enable offline conversations, download [Ollama](https://ollama.com) and run:
```bash
ollama run llama3
```

---

## 🧪 Intelligence Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend** | React, Vite, TypeScript, Lucide Icons, CSS3 |
| **Backend** | FastAPI, Python 3.x |
| **NLP** | NLTK, Keras (TensorFlow), BOW Model |
| **Offline AI** | Ollama (Llama3) |
| **Cloud AI** | Google Gemini 1.5 Flash |
| **Automation** | PyAutoGUI, Pycaw, PSUtil, Subprocess |

---

## 🙋‍♂️ Author

Developed and maintained by **Prince Joshi**  
*“Evolving from a simple bot to a professional neural workspace.”*

---

## 📄 License

Available under the [MIT License](LICENSE).
