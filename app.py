import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # ✅ Use TensorFlow backend

from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
import sqlite3
import numpy as np
import secrets
import requests
import logging
from keras.models import load_model  # ✅ Standalone Keras for compatibility
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

# ✅ Your custom loss
from model import focal_loss
from questions import get_random_questions

# ✅ Logging
logging.basicConfig(level=logging.DEBUG)

# ✅ Constants
MODEL_PATH = "alz_model.h5"
MODEL_URL = "https://huggingface.co/AnasHussain7/alz-model/resolve/main/alz_model.h5"

# ✅ Flask app setup
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# ✅ Login Manager
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
        flash("Invalid credentials", "danger")
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
            flash("Registration successful!", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists", "danger")
        finally:
            conn.close()
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
    return render_template('questionnaire.html', questions=get_random_questions())

@app.route('/result')
@login_required
def result():
    score = session.get('score', 0)
    if score >= 5:
        return redirect(url_for('upload'))
    return render_template('result.html', status="No significant signs detected", score=score)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'mri' not in request.files:
            return "No file uploaded"

        file = request.files['mri']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)

            try:
                if not os.path.exists(MODEL_PATH):
                    print("Downloading model from Hugging Face...")
                    with requests.get(MODEL_URL, stream=True) as r:
                        r.raise_for_status()
                        with open(MODEL_PATH, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    print("✅ Model downloaded successfully")

                img = load_img(filepath, target_size=(224, 224))
                img_array = img_to_array(img)
                img_array = preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)

                model = load_model(MODEL_PATH, custom_objects={'focal_loss_fixed': focal_loss()})
                prediction = model.predict(img_array)

                classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
                diagnosis = classes[np.argmax(prediction)]

                return render_template('result.html', status=diagnosis, score=session.get('score'))

            except Exception as e:
                print("Prediction Error:", e)
                return f"Internal Server Error: {str(e)}"

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
        flash("Message received (email not configured).", "success")
        return redirect(url_for('contact'))
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))



