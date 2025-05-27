#!/bin/bash

echo "🏥 Setting up AI Healthcare Platform v4.0..."

# Check prerequisites
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 required but not installed. Aborting." >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "❌ Docker required but not installed. Aborting." >&2; exit 1; }

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Setup environment
echo "⚙️ Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file from template"
    echo "⚠️ Please update .env with your API keys"
fi

# Initialize database
echo "🗄️ Initializing database..."
python -c "from app.database.db_manager import init_database; init_database()"

# Create directories
echo "📁 Creating directories..."
mkdir -p app/static/{dashboards,sample_data,models}
mkdir -p monitoring/grafana/dashboards
mkdir -p tests/{test_models,test_routes,test_utils}

# Download sample models (if needed)
echo "🤖 Setting up AI models..."
python -c "
import os
from transformers import AutoTokenizer, AutoModel
if os.getenv('HUGGINGFACE_TOKEN'):
    try:
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        print('✅ Downloaded embedding model')
    except:
        print('⚠️ Could not download models - check HuggingFace token')
"

echo "✅ Setup complete!"
echo "🚀 Next steps:"
echo "   1. Update .env with your API keys"
echo "   2. Run: docker-compose up --build"
echo "   3. Visit: http://localhost:8000/docs"
echo "   4. Monitor: http://localhost:3001 (Grafana)"
