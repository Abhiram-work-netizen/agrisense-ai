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
    print("⚠️ Warning: google-generativeai not installed. AI features will be disabled.")

# --- 2. PATH CONFIGURATION ---
# Get the absolute path of the current file (api/index.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define Root Dir (one level up from api/)
ROOT_DIR = os.path.join(current_dir, '..')

# Define standard paths for Flask
TEMPLATE_DIR = os.path.join(ROOT_DIR, 'templates')
STATIC_DIR = os.path.join(ROOT_DIR, 'static')

# Initialize Flask with explicit folder paths
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# --- 3. CONFIGURATION & KEYS ---
# It is best practice to use Environment Variables, but defaults are provided here for local testing.
GENAI_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCl1gjiOzdHKsJLpKQHblX2OqHWBE_i-ng") 
WEATHER_API_KEY = "43d15372405aeead0b442f9e0eac76aa"

if HAS_GENAI and GENAI_KEY:
    genai.configure(api_key=GENAI_KEY)

# --- 4. SMART MODEL LOADING ---
def load_file(filename):
    """
    Tries to load a file from multiple potential locations to handle 
    differences between Local Windows, Local Mac/Linux, and Vercel/Cloud paths.
    """
    possible_paths = [
        # 1. Inside api/models (Likely your local setup based on screenshot)
        os.path.join(current_dir, 'models', filename),
        # 2. In root/models (Standard Vercel structure)
        os.path.join(ROOT_DIR, 'models', filename),
        # 3. Direct relative path (Fallback)
        os.path.join('models', filename),
        # 4. Vercel Lambda absolute path
        os.path.join('/var/task/models', filename)
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                print(f"✅ Found {filename} at: {path}")
                with open(path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            # Just continue to the next path if this one fails
            continue
            
    print(f"❌ CRITICAL: Could not find {filename} in any known location.")
    return None

# Load models at startup
print("🌱 AgriSense Server Starting...")
model = load_file('model.pkl')
label_encoder = load_file('label_encoder.pkl')

# --- 5. ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if models loaded correctly
        if not model or not label_encoder:
            return jsonify({'success': False, 'error': "Server Error: AI Model not loaded. Check server logs."})

        # Parse input as float to ensure safety
        try:
            features = [
                float(request.form.get('N', 0)),
                float(request.form.get('P', 0)),
                float(request.form.get('K', 0)),
                float(request.form.get('temperature', 0)),
                float(request.form.get('humidity', 0)),
                float(request.form.get('ph', 0)),
                float(request.form.get('rainfall', 0))
            ]
        except ValueError:
            return jsonify({'success': False, 'error': "Invalid input: Please ensure all fields are numbers."})
        
        # Predict
        data_array = np.array([features])
        prediction_index = model.predict(data_array)
        result = label_encoder.inverse_transform(prediction_index)[0]
        
        return jsonify({'success': True, 'result': result})

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'success': False, 'error': f"Prediction Failed: {str(e)}"})

@app.route('/ask-ai', methods=['POST'])
def ask_ai():
    if not HAS_GENAI: 
        return jsonify({'answer': "AI Library Missing. Please install google-generativeai."})
    
    data = request.json
    user_question = data.get('question', '')
    image_data = data.get('image')
    
    try:
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Construct the prompt
        inputs = ["You are 'AgriVision Soil Doctor'. Answer agricultural questions concisely and helpfully.", f"Question: {user_question}"]
        
        # Add image if present
        if image_data:
            # Note: Ensure image_data is processed correctly (base64 decoding might be needed depending on frontend implementation)
            # For this code, we assume the library handles the raw passed data or you might need a small fix here if passing base64 string.
            inputs.append({"mime_type": "image/jpeg", "data": image_data})

        response = gemini_model.generate_content(inputs)
        return jsonify({'answer': response.text})
    except Exception as e:
        print(f"AI Error: {e}")
        return jsonify({'answer': "I'm having trouble connecting to the AI brain right now. Please try again."})

@app.route('/get-weather', methods=['POST'])
def get_weather():
    try:
        data = request.json
        lat, lon, city = data.get('lat'), data.get('lon'), data.get('city')
        
        if city:
            url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_API_KEY}&units=metric"
        else:
            url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        
        response = requests.get(url)
        if response.status_code != 200: 
            return jsonify({'success': False, 'error': 'Weather API Error'})
        
        weather_data = response.json()
        forecast_list = []
        seen_dates = set()
        
        # Process forecast data
        for item in weather_data['list']:
            date_txt = item['dt_txt'].split(' ')[0]
            if date_txt not in seen_dates:
                seen_dates.add(date_txt)
                forecast_list.append({
                    'day': date_txt,
                    'temp': round(item['main']['temp']),
                    'desc': item['weather'][0]['main'],
                    'icon_code': item['weather'][0]['icon'],
                    'rain_chance': int(item.get('pop', 0) * 100)
                })
                if len(forecast_list) >= 4: break
        
        current = weather_data['list'][0]
        return jsonify({
            'success': True, 
            'city': weather_data['city']['name'], 
            'forecast': forecast_list, 
            'current': {
                'humidity': current['main']['humidity'], 
                'wind_speed': current['wind']['speed'], 
                'pressure': current['main']['pressure'], 
                'rain_chance': int(current.get('pop', 0) * 100)
            }
        })
    except Exception as e:
        print(f"Weather Error: {e}")
        return jsonify({'success': False, 'error': str(e)})

# --- 6. LOCAL EXECUTION BLOCK (IMPORTANT) ---
# This block allows you to run 'python api/index.py' locally.
# It is ignored when deployed to Vercel (because Vercel imports 'app' directly).
if __name__ == '__main__':
    print("🚀 Starting Flask Server locally...")
    print(f"📂 Template Folder: {TEMPLATE_DIR}")
    print(f"📂 Static Folder: {STATIC_DIR}")
    # debug=True allows auto-reload when you change code
    app.run(debug=True, port=5000)