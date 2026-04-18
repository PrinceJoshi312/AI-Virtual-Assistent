# Software Requirements Specification (SRS)

## Project: Bugg AI Virtual Assistant

### 1. Introduction
Bugg AI is a cross-platform virtual assistant designed to simplify user interaction with their computer through natural language commands. It leverages NLP for intent recognition and LLMs for open-ended conversation.

### 2. System Overview
The application follows a client-server architecture:
- **Frontend:** A React-based web application providing a conversational interface.
- **Backend:** A FastAPI server that processes queries, manages intent classification, and executes system tasks.

### 3. Functional Requirements
- **R1: Query Processing** - The system shall process text and voice inputs.
- **R2: Intent Recognition** - The system shall identify user intents (e.g., "weather", "time", "screenshot") using a local classification model.
- **R3: AI Chat** - For intents not covered by local rules, the system shall use Google Gemini AI to generate responses.
- **R4: System Integration** - The system shall be capable of:
    - Locking the workstation.
    - Taking screenshots.
    - Controlling audio volume.
    - Monitoring CPU and RAM usage.
- **R5: Information Services** - The system shall provide real-time weather, time, and date information.
- **R6: Voice Output** - The system shall support Text-to-Speech for responses.

### 4. Non-Functional Requirements
- **Performance:** Local intent recognition should take less than 500ms.
- **Usability:** The interface should be intuitive and follow modern design principles.
- **Adaptability:** The UI must be mobile-responsive.

### 5. Directory Structure
```text
/
├── assets/         # Global assets
├── backend/        # FastAPI server, AI models, and logic
├── frontend/       # React application
├── scripts/        # Execution scripts (.bat, .sh)
└── SRS.md          # This document
```
