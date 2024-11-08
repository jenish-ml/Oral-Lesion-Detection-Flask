from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

model = load_model('models/model.h5')

img_height, img_width = 224, 224

class_labels = {
   0: 'Canker sore(Low Risk)',
    1: 'Cold sore(Low Risk)',
    2: 'Gum disease(Low Risk)',
    3:'Normal',
    4: 'Oral cancer(High Risk)',
    5: 'Oral thrush(Low Risk)'
}

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
@app.route('/')
def home():
    return render_template('register.html')
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'error')
            return redirect(url_for('register'))
        new_user = User(username=username, password=password) 
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:  # Check hashed password in real app
            login_user(user)
            return redirect(url_for('detection'))
        else:
            flash('Login unsuccessful. Check username and password', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/detection')
@login_required
def detection():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        file_path = os.path.join('temp', file.filename)
        file.save(file_path)

        img = image.load_img(file_path, target_size=(img_height, img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)

        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_label = class_labels.get(predicted_class_index, "Unknown")

        return jsonify({"prediction": predicted_class_label})

if __name__ == '__main__':
    os.makedirs('temp', exist_ok=True)
    with app.app_context():
        db.create_all() 
    
    app.run(debug=True)
