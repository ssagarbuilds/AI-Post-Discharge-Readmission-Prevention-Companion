# AI Post-Discharge Readmission Prevention Companion

A bilingual (English/Spanish) AI-powered healthcare app that predicts readmission risk and generates personalized care plans for post-discharge patients.

## 🚀 Features

- 🔮 **AI Risk Prediction**: HAIM + XGBoost + GPT-4o hybrid model
- 📋 **Care Plan Generation**: Automated, personalized post-discharge instructions
- 🤖 **Bilingual Chatbot**: Medical Spanish terminology support
- 🏠 **SDOH Detection**: Social determinants extraction from discharge notes
- 🔒 **Privacy-First**: Local deployment, encrypted data at rest
- 🌐 **Bilingual UI**: Full English/Spanish interface

## 📦 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.9+
- Git

### 1. Setup Backend
cd backend
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt

### 2. Configure Environment
Create `backend/.env`:
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=sqlite:///./healthcare.db
ENCRYPTION_KEY=your_32_character_encryption_key_here

### 3. Start Backend
cd backend
source .venv/bin/activate
python -m uvicorn app.main:app --reload

### 4. Setup Frontend
cd frontend
npm install
npm start

### 5. Access Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 🐳 Docker Deployment
docker-compose up --build

## 💰 Cost Breakdown
- Development: $0 (open source)
- Deployment: $5-20/month (VPS)
- API Costs: $50-100/month (OpenAI GPT-4o)
- **Total**: ~$200-300 startup capital

## 🔒 Security & Compliance
- HIPAA/DPDP compliant
- AES-256 encryption at rest
- No cloud dependencies
- Local data processing

## 📄 License
MIT License
