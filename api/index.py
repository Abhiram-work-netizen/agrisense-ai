import os
import pickle
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify

# --- 1. ROBUST AI IMPORT ---
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    print("⚠️ Warning: google-generativeai not installed.")

# --- 2. PATH CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(current_dir, '..')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
TEMPLATE_DIR = os.path.join(ROOT_DIR, 'templates')
STATIC_DIR = os.path.join(ROOT_DIR, 'static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# --- 3. API KEYS ---
GENAI_KEY = "AIzaSyDtvpP9plv3rfUKm2fPmzAyj_UQhSScotw"
WEATHER_API_KEY = "43d15372405aeead0b442f9e0eac76aa"

if HAS_GENAI and GENAI_KEY:
    genai.configure(api_key=GENAI_KEY)

# --- 4. LOAD MODELS (INCLUDING SCALER!) ---
def load_file(filename):
    path = os.path.join(MODEL_DIR, filename)
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

model = load_file('model.pkl')
label_encoder = load_file('label_encoder.pkl')
scaler = load_file('scaler.pkl')  # <--- CRITICAL: LOAD THE SCALER

# --- Helper Function for Safe Data ---
def get_safe_float(key, default=0.0):
    val = None
    if request.content_type == 'application/json':
        val = request.json.get(key)
    else:
        val = request.form.get(key)
    if val is None or val == "": return default
    try:
        return float(val)
    except ValueError:
        return default

# --- 5. ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if vital models loaded
        if not model or not label_encoder:
            return jsonify({'success': False, 'error': "AI Models failed to load."})

        # 1. Get Raw Data
        features = [
            get_safe_float('N'),
            get_safe_float('P'),
            get_safe_float('K'),
            get_safe_float('temperature'),
            get_safe_float('humidity'),
            get_safe_float('ph'),
            get_safe_float('rainfall')
        ]
        
        # 2. Convert to Numpy Array
        data_array = np.array([features])

        # 3. APPLY SCALING (The Missing Step)
        # If we have a scaler, we MUST use it. 
        if scaler:
            try:
                final_input = scaler.transform(data_array)
            except Exception:
                # Fallback: Sometimes scaler expects different feature names. 
                # If transform fails, try using raw data (but log warning)
                print("⚠️ Warning: Scaler failed. Using raw data.")
                final_input = data_array
        else:
            final_input = data_array

        # 4. Predict
        prediction_index = model.predict(final_input)
        result = label_encoder.inverse_transform(prediction_index)[0]
        
        return jsonify({'success': True, 'result': result})

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'success': False, 'error': f"Input Error: {str(e)}"})

@app.route('/ask-ai', methods=['POST'])
def ask_ai():
    if not HAS_GENAI: return jsonify({'answer': "⚠️ AI Library Missing on Server."})
    
    data = request.json
    user_question = data.get('question', '')
    image_data = data.get('image') 
    
    try:
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        content_parts = ["You are 'AgriVision Soil Doctor'. Answer concisely.", f"User Question: {user_question}"]
        if image_data:
            content_parts.append({"mime_type": "image/jpeg", "data": image_data})

        response = gemini_model.generate_content(content_parts)
        return jsonify({'answer': response.text})
    except Exception as e:
        print(f"AI Error: {e}")
        return jsonify({'answer': "I'm having trouble connecting to the satellite. Please try again."})

@app.route('/get-weather', methods=['POST'])
def get_weather():
    try:
        data = request.json
        lat, lon, city = data.get('lat'), data.get('lon'), data.get('city')
        if city: url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_API_KEY}&units=metric"
        else: url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        
        response = requests.get(url)
        if response.status_code != 200: return jsonify({'success': False, 'error': "Weather API Error"})
        
        weather_data = response.json()
        forecast_list = []
        seen_dates = set()
        for item in weather_data['list']:
            date_txt = item['dt_txt'].split(' ')[0]
            if date_txt not in seen_dates:
                seen_dates.add(date_txt)
                forecast_list.append({
                    'day': date_txt, 'temp': round(item['main']['temp']),
                    'desc': item['weather'][0]['main'], 'icon_code': item['weather'][0]['icon'],
                    'rain_chance': int(item.get('pop', 0) * 100)
                })
                if len(forecast_list) >= 4: break
        
        current = weather_data['list'][0]
        return jsonify({'success': True, 'city': weather_data['city']['name'], 'forecast': forecast_list,
            'current': {'humidity': current['main']['humidity'], 'wind_speed': current['wind']['speed'], 
            'pressure': current['main']['pressure'], 'rain_chance': int(current.get('pop', 0) * 100)}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    print("Starting AgriSense Server...")
    app.run(debug=True)