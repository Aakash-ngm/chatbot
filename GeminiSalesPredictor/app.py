from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import pickle
import torch
from llama_index.llms.groq import Groq
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from data_preprocessing import DataPreprocessor
from dotenv import load_dotenv

# ==========================================================
# ✅ Load environment variables
# ==========================================================
load_dotenv()
print("✓ Environment variables loaded")

# Flask App
app = Flask(__name__)
CORS(app)

# ==========================================================
# ✅ Globals for Models & Data
# ==========================================================
preprocessor = None
data_df = None
models = {}
tokenizers = {}
model_results = {}

# ==========================================================
# ✅ Load PlayStation Sales Data
# ==========================================================
def load_data():
    global preprocessor, data_df
    try:
        csv_path = "attached_assets/PlayStation Sales and Metadata (PS3PS4PS5) (Oct 2025)_1762234789301.csv"
        preprocessor = DataPreprocessor(csv_path)
        data_df = preprocessor.process_all()
        print(f"✓ Loaded dataset with {len(data_df)} records")
    except Exception as e:
        print(f"⚠️ Error loading data: {e}")

# ==========================================================
# ✅ Load Trained ML Models
# ==========================================================
def load_models():
    global models, tokenizers, model_results
    try:
        if os.path.exists('./models/deberta_model'):
            tokenizers['DeBERTa'] = DebertaV2Tokenizer.from_pretrained('./models/deberta_model')
            models['DeBERTa'] = DebertaV2ForSequenceClassification.from_pretrained('./models/deberta_model')
            models['DeBERTa'].eval()
            print("✓ Loaded DeBERTa model")

        if os.path.exists('./models/xgboost_model.pkl'):
            with open('./models/xgboost_model.pkl', 'rb') as f:
                models['XGBoost'] = pickle.load(f)
            print("✓ Loaded XGBoost model")

        if os.path.exists('./models/random_forest_model.pkl'):
            with open('./models/random_forest_model.pkl', 'rb') as f:
                models['Random Forest'] = pickle.load(f)
            print("✓ Loaded Random Forest model")

        if os.path.exists('./models/model_comparison.csv'):
            model_results = pd.read_csv('./models/model_comparison.csv', index_col=0).to_dict('index')

        print(f"✓ Total models loaded: {len(models)}")
    except Exception as e:
        print(f"⚠️ Could not load models: {e}")
        print("Please run model_training.py first to train your models.")

# ==========================================================
# ✅ Configure Groq LLM
# ==========================================================
groq_llm = None
try:
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    if not api_key:
        raise ValueError("❌ GROQ_API_KEY missing. Add it in your .env file.")

    groq_llm = Groq(model=model_name, api_key=api_key)
    print(f"✓ Groq API configured with model: {model_name}")
except Exception as e:
    print("⚠️ Groq API configuration failed:", str(e))
    groq_llm = None

# ==========================================================
# ✅ Initialize data & models
# ==========================================================
load_data()
load_models()

# ==========================================================
# ✅ Flask Routes
# ==========================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    if data_df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    stats = {
        'total_games': len(data_df),
        'total_sales': float(data_df['Total Sales'].sum()),
        'avg_rating': float(data_df['rating'].mean()),
        'avg_metacritic': float(data_df['metacritic'].mean()),
        'consoles': {
            'PS3': int((data_df['Console'] == 'PS3').sum()),
            'PS4': int((data_df['Console'] == 'PS4').sum()),
            'PS5': int((data_df['Console'] == 'PS5').sum())
        },
        'top_publishers': data_df['Publisher'].value_counts().head(10).to_dict()
    }
    return jsonify(stats)

@app.route('/api/sales_data')
def get_sales_data():
    if data_df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    try:
        sales_by_console = data_df.groupby('Console')['Total Sales'].sum().to_dict()
        sales_by_year = data_df.groupby('Release Year')['Total Sales'].sum().sort_index().to_dict()
        top_games = data_df.nlargest(10, 'Total Sales')[['Name', 'Total Sales', 'Console']].to_dict('records')
        regional_sales = {
            'NA': float(data_df['NA Sales'].sum()),
            'PAL': float(data_df['PAL Sales'].sum()),
            'Japan': float(data_df['Japan Sales'].sum()),
            'Other': float(data_df['Other Sales'].sum())
        }
        genres = {}
        for _, row in data_df.iterrows():
            genre_list = str(row['genres']).split(',')
            for genre in genre_list:
                genre = genre.strip()
                if genre and genre != 'Unknown':
                    genres[genre] = genres.get(genre, 0) + 1
        return jsonify({
            'sales_by_console': sales_by_console,
            'sales_by_year': {str(int(k)): v for k, v in sales_by_year.items() if not pd.isna(k)},
            'top_games': top_games,
            'regional_sales': regional_sales,
            'genres': dict(sorted(genres.items(), key=lambda x: x[1], reverse=True)[:10])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==========================================================
# ✅ Short and Focused Chatbot Endpoint
# ==========================================================
@app.route('/api/chat', methods=['POST'])
def chat():
    """Chatbot endpoint using Groq API (short + focused responses)"""
    if groq_llm is None:
        return jsonify({
            'response': 'Groq API not configured. Add GROQ_API_KEY in your .env file.',
            'error': True
        })

    try:
        message = request.json.get('message', '').strip()

        # 🔹 Context prompt for short, factual answers
        context = f"""
        You are a PlayStation sales analysis assistant.
        Respond with only 3–5 short sentences.
        Focus only on important facts — sales, top games, console comparison, and rating insights.
        Avoid long explanations, opinions, or repeating context.

        User question: {message}
        """

        # 🔹 Get concise response
        response = groq_llm.complete(context)
        short_reply = response.text.strip()

        # 🔹 Auto-trim long replies (>80 words)
        if len(short_reply.split()) > 80:
            short_reply = " ".join(short_reply.split()[:80]) + "..."

        return jsonify({'response': short_reply, 'error': False})

    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}', 'error': True})

@app.route('/api/groq_status')
def groq_status():
    configured = groq_llm is not None
    return jsonify({'configured': configured, 'message': 'Groq API ready' if configured else 'Groq API not configured'})

# ==========================================================
# ✅ Run Flask App
# ==========================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎮 PLAYSTATION SALES PREDICTION SYSTEM (Groq Version)")
    print("="*60)
    if groq_llm:
        print(f"✓ Groq model active: {os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}")
    else:
        print("⚠️ Groq not configured — check your .env setup")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
