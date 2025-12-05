import os
import pickle
import numpy as np
import requests
import json
from flask import Flask, render_template, request, jsonify

# --- ROBUST IMPORT: AI LIBRARY ---
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    print("⚠️ Warning: google-generativeai not installed. AI Chat will be disabled.")

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# API Keys
GENAI_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCl1gjiOzdHKsJLpKQHblX2OqHWBE_i-ng") 
WEATHER_API_KEY = "43d15372405aeead0b442f9e0eac76aa"

if HAS_GENAI and GENAI_KEY:
    genai.configure(api_key=GENAI_KEY)

# --- LOAD MODELS ---
def load_file(filename):
    path = os.path.join(MODEL_DIR, filename)
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return None

model = load_file('model.pkl')
label_encoder = load_file('label_encoder.pkl')

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form.get('N')),
            float(request.form.get('P')),
            float(request.form.get('K')),
            float(request.form.get('temperature')),
            float(request.form.get('humidity')),
            float(request.form.get('ph')),
            float(request.form.get('rainfall'))
        ]
        
        if model and label_encoder:
            data_array = np.array([features])
            prediction_index = model.predict(data_array)
            result = label_encoder.inverse_transform(prediction_index)[0]
            return jsonify({'success': True, 'result': result})
        else:
            return jsonify({'success': False, 'error': "AI Model files are missing from the server."})

    except Exception as e:
        return jsonify({'success': False, 'error': f"Input Error: {str(e)}"})

@app.route('/ask-ai', methods=['POST'])
def ask_ai():
    if not HAS_GENAI: return jsonify({'answer': "⚠️ AI Library Missing."})
    if not GENAI_KEY: return jsonify({'answer': "⚠️ API Key Missing."})
    
    data = request.json
    user_question = data.get('question', '')
    image_data = data.get('image') # Base64 string if present
    
    try:
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Prepare inputs
        inputs = []
        
        # System Prompt logic embedded in user prompt for better context
        prompt_text = (
            "You are 'AgriVision Soil Doctor', an expert agricultural AI. "
            "Analyze the crop/soil images if provided. Answer farming questions concisely. "
            f"User Question: {user_question}"
        )
        inputs.append(prompt_text)

        if image_data:
            # Create the image part structure for Gemini
            image_part = {
                "mime_type": "image/jpeg",
                "data": image_data
            }
            inputs.append(image_part)

        response = gemini_model.generate_content(inputs)
        return jsonify({'answer': response.text})
    except Exception as e:
        print(f"AI Error: {e}")
        return jsonify({'answer': "I'm having trouble analyzing that. Please try again."})

@app.route('/get-weather', methods=['POST'])
def get_weather():
    """Fetches real weather + Rainfall Probability"""
    try:
        data = request.json
        lat = data.get('lat')
        lon = data.get('lon')
        city = data.get('city')
        
        if city:
            url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_API_KEY}&units=metric"
        else:
            url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        
        response = requests.get(url)
        weather_data = response.json()
        
        if response.status_code != 200:
            return jsonify({'success': False, 'error': weather_data.get('message', 'Weather API Error')})

        forecast_list = []
        seen_dates = set()
        
        for item in weather_data['list']:
            date_txt = item['dt_txt'].split(' ')[0]
            if date_txt not in seen_dates:
                seen_dates.add(date_txt)
                rain_chance = int(item.get('pop', 0) * 100)
                
                forecast_list.append({
                    'day': date_txt,
                    'temp': round(item['main']['temp']),
                    'desc': item['weather'][0]['main'],
                    'icon_code': item['weather'][0]['icon'],
                    'rain_chance': rain_chance
                })
                if len(forecast_list) >= 4: break
        
        current = weather_data['list'][0]
        current_pop = int(current.get('pop', 0) * 100)
        
        current_details = {
            'humidity': current['main']['humidity'],
            'wind_speed': current['wind']['speed'],
            'pressure': current['main']['pressure'],
            'rain_chance': current_pop
        }
        
        return jsonify({
            'success': True, 
            'city': weather_data['city']['name'], 
            'forecast': forecast_list,
            'current': current_details
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)