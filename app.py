# Importing necessary libraries
import warnings
from flask import redirect, url_for, flash
from flask import Flask, render_template, request, session
from flask_sqlalchemy import SQLAlchemy
from predictor import Predictor

# Suppressing warnings
warnings.filterwarnings("ignore")

# Creating Flask application
app = Flask(__name__)
app.secret_key = 'abcde'

# SQLAlchemy configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/login_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Creating SQLAlchemy object
db = SQLAlchemy(app)

# Defining User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Checking database connection
with app.app_context():
    try:
        db.create_all()  # Try to create all tables based on defined models
        print("Connected to database successfully!")
    except Exception as e:
        print("Failed to connect to database:", e)

# Defining routes
@app.route('/predict')
def home():
        return render_template('index.html')
    

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Query the database for a user with the entered email
        user = User.query.filter_by(email=email).first()
        if user:
            # Check if the entered password matches the stored password
            if password == user.password:
                session['email'] = user.email
                return render_template('index.html')  # Redirect to the desired page upon successful login
        # If no user with the entered email or incorrect password, render the login page with an error message
        return render_template('homepage.html', error='Invalid email or password')
    return render_template('homepage.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm')
        if password != confirm:
            flash('Passwords does not match. Please try again.')
            return redirect(url_for('signup'))
        else:
            entry = User(email=email, password=password)
            db.session.add(entry)
            db.session.commit()
    return render_template('homepage.html')

@app.route('/search', methods=['GET'])
def search():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    company = request.form['company']
    days = int(request.form['days'])

    predictor = Predictor(company, days)
    accuracy = predictor.calculate_accuracy()
    long_name, predprice, current_price, plot_filename, headlines = predictor.get_all_data_for_web()
    return render_template('result.html', long_name=long_name, company=company, rmse=accuracy, prediction=predprice, current_price=current_price, days=days, plot_url=plot_filename, headlines=headlines)

if __name__ == '__main__':
    app.run(debug=True)
