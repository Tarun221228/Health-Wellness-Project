import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import json
from datetime import datetime, timedelta
import random
import requests

app = Flask(__name__)
app.secret_key = "healthcare_secret_key_2023"
app.config['UPLOAD_FOLDER'] = 'static/images'

# SQLite Database Connection
def get_db_connection():
    conn = sqlite3.connect('healthcare.db')
    conn.row_factory = sqlite3.Row  # This enables name-based access to columns
    return conn

# Load Machine Learning Models with improved error handling
models = {}
model_dir = "models"

if os.path.exists(model_dir):
    for fname in os.listdir(model_dir):
        if fname.endswith(".pkl") or fname.endswith(".joblib"):
            path = os.path.join(model_dir, fname)
            try:
                # Try loading with joblib first
                import joblib
                model_name = fname.replace(".pkl", "").replace(".joblib", "")
                models[model_name] = joblib.load(path)
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading model {fname} with joblib: {str(e)}")
                try:
                    # Fallback to pickle with specific encoding
                    with open(path, "rb") as f:
                        models[model_name] = pickle.load(f, encoding='latin1')
                    print(f"Loaded model {model_name} with pickle (latin1 encoding)")
                except Exception as e2:
                    print(f"Error loading model {fname} with pickle: {str(e2)}")

print("Loaded models:", list(models.keys()))

# Health tips data
health_tips = [
    {
        "title": "Stay Hydrated",
        "content": "Drink at least 8 glasses of water daily to maintain optimal body function.",
        "category": "Nutrition",
        "imageUrl": "/static/images/water-glass.png"
    },
    {
        "title": "Regular Exercise",
        "content": "Aim for at least 30 minutes of moderate exercise most days of the week.",
        "category": "Fitness",
        "imageUrl": "/static/images/exercise.png"
    },
    {
        "title": "Balanced Diet",
        "content": "Include fruits, vegetables, lean proteins, and whole grains in your daily meals.",
        "category": "Nutrition",
        "imageUrl": "/static/images/balanced-diet.png"
    },
    {
        "title": "Adequate Sleep",
        "content": "Most adults need 7-9 hours of quality sleep each night for optimal health.",
        "category": "Wellness",
        "imageUrl": "/static/images/sleep.png"
    }
]

# Chat history storage (in-memory for demo, should be in database for production)
chat_history = {}

# Debug route for database testing
@app.route("/debug/database")
def debug_database():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if predictions table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
        predictions_table = cursor.fetchone()
        
        # Count predictions
        cursor.execute("SELECT COUNT(*) as count FROM predictions")
        predictions_count = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            "predictions_table_exists": bool(predictions_table),
            "predictions_count": predictions_count
        })
    except Exception as e:
        return f"Database error: {str(e)}"

# API ENDPOINTS

@app.route("/api/dashboard-data")
def api_dashboard_data():
    # Anonymous access
    user_id = 1
    username = "Guest"
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get recent predictions
    cursor.execute("""
        SELECT prediction, medication, dosage, created_at 
        FROM predictions 
        WHERE user_id=? 
        ORDER BY created_at DESC 
        LIMIT 5
    """, (user_id,))
    recent_predictions = cursor.fetchall()
    
    # Convert to list of dictionaries for JSON serialization
    predictions_list = []
    for pred in recent_predictions:
        predictions_list.append({
            "prediction": pred["prediction"],
            "medication": pred["medication"],
            "dosage": pred["dosage"],
            "created_at": pred["created_at"]
        })
    
    # Generate sample health data
    health_score = 85
    steps_today = random.randint(6000, 10000)
    water_intake = random.randint(4, 8)
    sleep_hours = round(random.uniform(6.5, 8.5), 1)
    
    conn.close()
    
    return jsonify({
        "health_score": health_score,
        "steps_today": steps_today,
        "water_intake": water_intake,
        "sleep_hours": sleep_hours,
        "recent_predictions": predictions_list,
        "username": username
    })

@app.route("/api/health-trends")
def api_health_trends():
    # Sample health trends data (anonymous access)
    trends_data = {
        "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "health_scores": [72, 75, 78, 80, 82, 85],
        "steps": [6500, 7200, 6800, 7500, 8000, 7800],
        "sleep_hours": [6.5, 7.0, 7.2, 7.5, 7.8, 8.0]
    }
    
    return jsonify(trends_data)

@app.route("/api/health-history")
def api_health_history():
    # Anonymous access
    user_id = 1
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT prediction, medication, dosage, created_at 
        FROM predictions 
        WHERE user_id=? 
        ORDER BY created_at DESC
    """, (user_id,))
    history = cursor.fetchall()
    
    history_list = []
    for record in history:
        history_list.append({
            "prediction": record["prediction"],
            "medication": record["medication"],
            "dosage": record["dosage"],
            "created_at": record["created_at"]
        })
    
    conn.close()
    
    return jsonify(history_list)

@app.route("/api/health-tips")
def api_health_tips():
    return jsonify(health_tips)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        user_id = data.get('user_id', 'guest')  # Default to guest for anonymous access

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        # Initialize chat history for new users
        if user_id not in chat_history:
            chat_history[user_id] = []

        # Add user message to history
        chat_history[user_id].append({"role": "user", "content": user_message})

        # Prepare context from recent chat history (last 10 messages)
        context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in chat_history[user_id][-10:]
        ])

        # Create prompt for Google Gemini (based on workflow structure)
        prompt = f"""You are a helpful healthcare assistant. Provide accurate, helpful responses about health and wellness.

Context from previous conversation:
{context}

Current user question: {user_message}

Please provide a helpful, accurate response about health and wellness. If this involves medical advice, remind the user to consult healthcare professionals."""

        # Call Google Gemini API (you'll need to set up API key)
        # For now, return a mock response - replace with actual API call
        response_text = generate_mock_gemini_response(user_message, context)

        # Add AI response to history
        chat_history[user_id].append({"role": "assistant", "content": response_text})

        return jsonify({
            "response": response_text,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Chat API error: {str(e)}")
        return jsonify({"error": "Failed to process chat message"}), 500

def generate_mock_gemini_response(user_message, context):
    """Mock Gemini response - replace with actual API call"""
    # Simple keyword-based responses for demo
    message_lower = user_message.lower()

    if "headache" in message_lower:
        return "Headaches can have many causes including stress, dehydration, or lack of sleep. I recommend staying hydrated, getting adequate rest, and if headaches persist, consulting a healthcare professional."
    elif "exercise" in message_lower or "workout" in message_lower:
        return "Regular exercise is great for health! Aim for at least 150 minutes of moderate aerobic activity per week. Start slowly and consult your doctor before beginning a new exercise program."
    elif "diet" in message_lower or "nutrition" in message_lower or "food" in message_lower:
        return "A balanced diet rich in fruits, vegetables, whole grains, and lean proteins is essential for good health. Consider consulting a registered dietitian for personalized nutrition advice."
    elif "sleep" in message_lower or "rest" in message_lower:
        return "Most adults need 7-9 hours of quality sleep per night. Good sleep hygiene includes maintaining a consistent schedule, creating a dark and quiet sleep environment, and avoiding screens before bedtime."
    elif "test" in message_lower or "hi" in message_lower or "hello" in message_lower:
        return "Hello! I'm your AI health assistant. I can help with wellness questions. Try asking about exercise, diet, sleep, or headaches for specific advice."
    elif "health" in message_lower or "wellness" in message_lower:
        return "Maintaining good health involves balanced nutrition, regular exercise, adequate sleep, and stress management. Remember to consult professionals for medical concerns."
    else:
        # Varied default responses to avoid repetition
        defaults = [
            "I'm here to help with your health and wellness questions. For personalized medical advice, please consult with a qualified healthcare professional. What specific topic would you like to discuss?",
            "Health is a journey! Let's talk about nutrition, fitness, or sleep. What's on your mind today?",
            "As your AI assistant, I can provide general wellness tips. For any symptoms, see a doctor. How can I assist?"
        ]
        import random
        return random.choice(defaults)
            
# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        subject = request.form["subject"]
        message = request.form["message"]
        
        # Here you would typically save this to a database
        print(f"Contact form submission: {name}, {email}, {subject}, {message}")
        return redirect(url_for("contact"))
    
    return render_template("contact.html")

@app.route("/dashboard")
def dashboard():
    # Anonymous access
    user_id = 1
    username = "Guest"
    
    # Get prediction history
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT prediction, medication, dosage, created_at 
        FROM predictions 
        WHERE user_id=? 
        ORDER BY created_at DESC 
        LIMIT 5
    """, (user_id,))
    recent_predictions = cursor.fetchall()
    
    # Get health stats for dashboard
    cursor.execute("""
        SELECT 
            COUNT(*) as total_predictions,
            MAX(created_at) as last_prediction,
            (SELECT prediction FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 1) as latest_condition
        FROM predictions 
        WHERE user_id=?
    """, (user_id, user_id))
    stats = cursor.fetchone()
    
    conn.close()
    
    # Generate some sample health data for demonstration
    health_score = 85  # This would be calculated based on user's health data
    steps_today = random.randint(6000, 10000)
    water_intake = random.randint(4, 8)
    sleep_hours = round(random.uniform(6.5, 8.5), 1)
    
    return render_template("dashboard.html", 
                         username=username,
                         recent_predictions=recent_predictions,
                         stats=stats,
                         health_score=health_score,
                         steps_today=steps_today,
                         water_intake=water_intake,
                         sleep_hours=sleep_hours)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Use dummy user_id for anonymous access
    user_id = 1

    if request.method == "POST":
        try:
            # Get form data with correct feature columns (ignore gender if sent)
            age = int(request.form["age"])
            height_cm = float(request.form.get("height_cm", 0))
            weight_kg = float(request.form.get("weight_kg", 0))
            systolic_bp = float(request.form.get("systolic_bp", 0))
            diastolic_bp = float(request.form.get("diastolic_bp", 0))
            heart_rate = float(request.form.get("heart_rate", 0))
            temperature = float(request.form.get("temperature", 0))
            
            # Create feature vector based on your dataset structure (no gender)
            features = [[
                age, 
                height_cm, 
                weight_kg, 
                systolic_bp, 
                diastolic_bp, 
                heart_rate, 
                temperature
            ]]
            
            # Use RandomForest_disease model
            if "RandomForest_disease" in models:
                model = models["RandomForest_disease"]
                prediction = model.predict(features)[0]
                
                # For medication and dosage
                medication = "Unknown"
                dosage = "Consult doctor"
                
                if "RandomForest_medication_name" in models:
                    medication_model = models["RandomForest_medication_name"]
                    medication = medication_model.predict(features)[0]
                
                if "RandomForest_dosage" in models:
                    dosage_model = models["RandomForest_dosage"]
                    dosage = dosage_model.predict(features)[0]
                
                # Save prediction to database
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO predictions (user_id, age, height_cm, weight_kg, systolic_bp, diastolic_bp, heart_rate, temperature, prediction, medication, dosage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_id, age, height_cm, weight_kg, systolic_bp, diastolic_bp, heart_rate, temperature, prediction, medication, dosage))
                conn.commit()
                conn.close()
                
                # Return JSON response for AJAX requests
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        "success": True,
                        "prediction": prediction,
                        "medication": medication,
                        "dosage": dosage
                    })
                
                return render_template("result.html", 
                                     prediction=prediction, 
                                     medication=medication,
                                     dosage=dosage)
            else:
                # Fallback if models aren't loaded
                result = {
                    "prediction": "Common Cold", 
                    "medication": "Antihistamines",
                    "dosage": "As directed"
                }
                
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({"success": True, **result})
                
                return render_template("result.html", **result)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"success": False, "error": error_msg})
            
            return render_template("result.html", 
                                 prediction=error_msg, 
                                 medication="N/A",
                                 dosage="N/A")

    return render_template("index.html")

# Serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Initialize database tables
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create predictions table (no users table, no FK)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            age INTEGER,
            height_cm REAL,
            weight_kg REAL,
            systolic_bp REAL,
            diastolic_bp REAL,
            heart_rate REAL,
            temperature REAL,
            prediction TEXT,
            medication TEXT,
            dosage TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

# Error handler to return JSON for API routes
@app.errorhandler(Exception)
def handle_exception(e):
    if request.path.startswith('/api/') or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({"error": str(e)}), 500
    # For non-API routes, let Flask handle with HTML
    return render_template('error.html', error=str(e)), 500

@app.errorhandler(404)
def handle_404(e):
    if request.path.startswith('/api/') or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({"error": "Not found"}), 404
    return render_template('404.html'), 404

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
