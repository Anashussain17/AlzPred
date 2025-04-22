from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import sqlite3
import os
import tensorflow
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from questions import get_random_questions
from tensorflow.keras.applications.resnet50 import preprocess_input
import secrets
import gdown
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Model config
MODEL_PATH = "alz_model.keras"
MODEL_URL = "https://drive.google.com/file/d/1y-kMJGWLci87bv7v4mizNsjvr2RvS2U3/view?usp=sharing" # Direct download URL

# Download and verify model
if not os.path.exists(MODEL_PATH):
    print("🔄 Downloading model from Google Drive...")
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        # Verify model integrity
        try:
            load_model(MODEL_PATH)
            print("✅ Model verified and loaded successfully.")
        except Exception as e:
            print(f"❌ Corrupted model file: {str(e)}")
            os.remove(MODEL_PATH)
            exit(1)
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        exit(1)

# App config
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    pass

@login_manager.user_loader
def load_user(user_id):
    user = User()
    user.id = user_id
    return user

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and user['password'] == password:
            user_obj = User()
            user_obj.id = username
            login_user(user_obj)
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))

        flash("Invalid username or password", "danger")
        return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
            flash("Registration successful! You can now log in.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists. Please choose a different one.", "danger")
        finally:
            conn.close()
        return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/questionnaire', methods=['GET', 'POST'])
@login_required
def questionnaire():
    if request.method == 'POST':
        score = sum(1 for key in request.form if request.form[key] == 'yes')
        session['score'] = score
        return redirect(url_for('result'))
    questions = get_random_questions()
    return render_template('questionnaire.html', questions=questions)

@app.route('/result')
@login_required
def result():
    score = session.get('score', 0)
    if score >= 5:
        return redirect(url_for('upload'))
    return render_template('result.html', status='No significant signs detected', score=score)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'mri' not in request.files:
            flash("No file uploaded", "danger")
            return redirect(request.url)
        
        file = request.files['mri']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Load and process image
                img = load_img(filepath, target_size=(224, 224))
                img_array = img_to_array(img)
                img_array = preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)

                # Load model and predict
                try:
                    model = load_model(MODEL_PATH)
                except Exception as e:
                    logging.error(f"Model loading failed: {str(e)}")
                    flash("Diagnosis service temporarily unavailable", "danger")
                    return redirect(url_for('dashboard'))

                prediction = model.predict(img_array)
                classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
                diagnosis = classes[np.argmax(prediction)]

                return render_template('result.html', status=diagnosis, score=session.get('score'))
            except Exception as e:
                logging.error(f"Prediction error: {str(e)}")
                flash("Error processing MRI scan", "danger")
                return redirect(url_for('upload'))
            finally:
                # Cleanup uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)

    return render_template('upload.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/howitworks')
def how_it_works():
    return render_template('howitworks.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Implement your email sending logic here
        flash("Message received! We'll respond within 24 hours.", "success")
        return redirect(url_for('contact'))
    return render_template('contact.html')

# Startup configuration
if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))


