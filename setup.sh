#!/bin/bash

echo "🏥 Setting up AI Post-Discharge Readmission Prevention Companion..."

# Check prerequisites
command -v node >/dev/null 2>&1 || { echo "❌ Node.js required but not installed. Aborting." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 required but not installed. Aborting." >&2; exit 1; }

# Backend setup
echo "📦 Setting up backend..."
cd backend || exit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "from app.database.db_manager import init_database; init_database()"
cd ..

# Frontend setup
echo "🎨 Setting up frontend..."
cd frontend || exit
npm install
cd ..

# Create environment files
echo "⚙️ Creating environment files..."
cat > backend/.env << EOF
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=sqlite:///./healthcare.db
ENCRYPTION_KEY=your_32_character_encryption_key_here
EOF

cat > frontend/.env << EOF
REACT_APP_API_URL=http://localhost:8000
EOF

echo "✅ Setup complete!"
echo "📝 Next steps:"
echo "   1. Add your OpenAI API key to backend/.env"
echo "   2. Generate a secure encryption key for backend/.env"
echo "   3. Run 'docker-compose up' or start backend/frontend separately"
