from flask import Flask, render_template, request, redirect, url_for, session,flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user,current_user
import sqlite3
import os, tensorflow
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from questions import get_random_questions
from tensorflow.keras.applications.resnet50 import preprocess_input 
import secrets
from flask import current_app


app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

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
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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
    return render_template('result.html', 
                         status='No significant signs detected',
                         score=score)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'mri' not in request.files:
            return 'No file uploaded'
        
        file = request.files['mri']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img) 
            img_array = preprocess_input(img_array) 
            img_array = np.expand_dims(img_array, axis=0)
            
            model = load_model("alz_model.keras")
            prediction = model.predict(img_array)
            classes = ['MildDemented', 'ModerateDemented', 
                      'NonDemented', 'VeryMildDemented']
            diagnosis = classes[np.argmax(prediction)]
            
            return render_template('result.html', 
                                 status=diagnosis,
                                 score=session.get('score'))
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
        name = request.form.get('name')
        email = request.form.get('email')
        message_text = request.form.get('message')

        msg = Message(
            subject=f"New Contact Form Submission from {name}",
            sender=email, 
            recipients=["alzpredict66@gmail.com"], 
            reply_to=email  
        )
        msg.body = f"Message from {name} ({email}):\n\n{message_text}"
        try:
            mail.send(msg)
            flash("Your message has been sent successfully!", "success")
        except Exception as e:
            flash("An error occurred while sending your message. Please try again later.", "danger")
            print("Mail send error:", e)
        return redirect(url_for('contact'))
    return render_template('contact.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)

