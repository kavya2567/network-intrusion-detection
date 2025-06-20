# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import io
import base64

# ML and Visualization Libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
# IMPORTANT: Change this to a strong, random key in production!
app.secret_key = 'your_super_secret_key_12345'

# --- File Upload Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Database setup ---
DATABASE = 'users.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize the database when the application starts
with app.app_context():
    init_db()

# --- Utility function to convert Matplotlib figures to base64 images ---
def get_image_base64(fig):
    """Converts a Matplotlib figure to a base64 encoded PNG string."""
    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format='png', bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    img_bytes.seek(0)
    return base64.b64encode(img_bytes.getvalue()).decode()

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        # In a real app, hash this password using werkzeug.security!
        password = request.form['password'] 

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists. Please choose a different one.', 'danger')
        finally:
            conn.close()
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        # In a real app, use check_password_hash() here!
        user = cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password)).fetchone()
        conn.close()

        if user:
            session['logged_in'] = True
            session['username'] = user['username']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        flash('Please log in to access the dashboard.', 'warning')
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/input_data', methods=['GET', 'POST'])
def input_data():
    if not session.get('logged_in'):
        flash('Please log in to access this feature.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                flash('No selected file', 'warning')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                session['uploaded_file'] = filepath # Store path in session for ML analysis
                flash(f'File "{filename}" uploaded successfully! You can now run ML analysis.', 'success')
                return redirect(url_for('run_ml_analysis'))
            else:
                flash('Invalid file type. Only CSV files are allowed.', 'danger')
                return redirect(request.url)
        elif 'generate_data' in request.form:
            # Generate a more realistic dummy dataset for demonstration
            generated_data_filename = 'generated_intrusion_data.csv'
            generated_data_path = os.path.join(app.config['UPLOAD_FOLDER'], generated_data_filename)
            
            # This dummy data mimics some features found in NSL-KDD
            data = {
                'duration': np.random.randint(0, 100, 50),
                'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], 50, p=[0.7, 0.2, 0.1]),
                'service': np.random.choice(['http', 'smtp', 'ftp', 'dns', 'ecr_i', 'finger'], 50),
                'src_bytes': np.random.randint(0, 1000, 50),
                'dst_bytes': np.random.randint(0, 2000, 50),
                'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTO'], 50),
                'logged_in': np.random.choice([0, 1], 50, p=[0.6, 0.4]),
                'num_compromised': np.random.randint(0, 5, 50),
                'dst_host_count': np.random.randint(1, 255, 50),
                'same_srv_rate': np.random.rand(50),
                'attack_type': np.random.choice(
                    ['normal', 'dos', 'probe', 'r2l', 'u2r'], 50,
                    p=[0.6, 0.2, 0.1, 0.05, 0.05] # Imbalance for demonstration
                )
            }
            pd.DataFrame(data).to_csv(generated_data_path, index=False)
            
            session['uploaded_file'] = generated_data_path # Store path
            flash(f'Synthetic data ({generated_data_filename}) generated successfully! You can now run ML analysis.', 'success')
            return redirect(url_for('run_ml_analysis'))

    return render_template('input_data.html')

@app.route('/run_ml_analysis')
def run_ml_analysis():
    if not session.get('logged_in'):
        flash('Please log in to access this feature.', 'warning')
        return redirect(url_for('login'))

    data_filepath = session.get('uploaded_file')
    if not data_filepath or not os.path.exists(data_filepath):
        flash('No data uploaded or generated yet. Please upload or generate data first.', 'warning')
        return redirect(url_for('input_data'))

    ml_report_summary = ""
    plot_images = {} # Dictionary to store base64 plot images

    try:
        df = pd.read_csv(data_filepath)
        
        ml_report_summary += f"Successfully loaded data from: **{os.path.basename(data_filepath)}**\n"
        ml_report_summary += f"Dataset rows: {len(df)}, columns: {len(df.columns)}\n"
        ml_report_summary += f"First 5 rows:\n{df.head().to_string()}\n\n"
        
        # --- Data Preprocessing ---
        if 'attack_type' not in df.columns:
            flash("The dataset must contain an 'attack_type' column for intrusion detection.", "danger")
            return render_template('ml_reports.html', ml_summary="Error: 'attack_type' column missing.", plot_images={})

        X = df.drop('attack_type', axis=1)
        y = df['attack_type']

        # Check and report original class distribution (Solution 1)
        ml_report_summary += f"\n--- Original Class Distribution in Dataset ---\n"
        class_counts_orig = y.value_counts()
        ml_report_summary += f"{class_counts_orig.to_string()}\n"
        ml_report_summary += f"Smallest class count: **{class_counts_orig.min()}**\n\n"

        # Optional: Plot original class distribution
        fig_class_dist_orig = plt.figure(figsize=(10, 6))
        sns.countplot(x=y, palette='viridis', order=class_counts_orig.index)
        plt.title('Distribution of Target Classes (Original)')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plot_images['original_class_distribution'] = get_image_base64(fig_class_dist_orig)

        # --- Handle Rare Classes (Solution 4 - Grouping) ---
        # Define a threshold for rare classes (e.g., less than 5 instances)
        RARE_CLASS_THRESHOLD = 5
        rare_classes = class_counts_orig[class_counts_orig < RARE_CLASS_THRESHOLD].index

        if not rare_classes.empty:
            ml_report_summary += f"ACTION: Grouping rare classes (count < {RARE_CLASS_THRESHOLD}): {rare_classes.tolist()} into 'rare_attack'\n"
            y_processed = y.apply(lambda x: 'rare_attack' if x in rare_classes else x)
            ml_report_summary += f"New class distribution after grouping:\n{y_processed.value_counts().to_string()}\n\n"
            y = y_processed # Use the modified target variable for splitting and training
        
        # Determine if stratification is possible and necessary
        # Stratify can only be used if all classes in y have at least 2 samples
        can_stratify = all(count >= 2 for count in y.value_counts()) and len(y.unique()) > 1

        # Handle categorical features (Label Encoding for simplicity with tree models)
        categorical_cols = X.select_dtypes(include='object').columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        # Handle numerical features (Scaling)
        numerical_cols = X.select_dtypes(include=np.number).columns
        if not numerical_cols.empty:
            scaler = StandardScaler()
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

        # Handle any remaining NaNs (simple imputation for demo)
        X = X.fillna(0)

        # Ensure enough samples to split
        if len(X) < 2:
            flash("Not enough data to perform train-test split for ML model training. Please upload or generate more data.", "warning")
            return render_template('ml_reports.html', ml_summary="Error: Insufficient data for ML.", plot_images={})

        # --- Train-Test Split (Solution 3 - Adjusted test_size) ---
        test_size = 0.25 # Reduced from 0.3 to increase training data

        split_params = {
            'test_size': test_size,
            'random_state': 42
        }
        if can_stratify:
            split_params['stratify'] = y
            ml_report_summary += f"Performing **stratified split** (test_size={test_size})...\n"
        else:
            ml_report_summary += f"**WARNING**: Cannot perform stratified split (test_size={test_size}) due to classes with < 2 samples. Proceeding without stratification.\n"
            flash("Warning: Not enough samples in some classes for stratified split. Results may be less reliable.", "warning")

        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_params)

        ml_report_summary += f"\n--- ML Model Training and Evaluation ---\n"
        ml_report_summary += f"Train set size: {len(X_train)} samples\n"
        ml_report_summary += f"Test set size: {len(X_test)} samples\n"
        ml_report_summary += f"Train set class distribution:\n{y_train.value_counts().to_string()}\n"
        ml_report_summary += f"Test set class distribution:\n{y_test.value_counts().to_string()}\n\n"
        
        # --- Decision Tree Classifier ---
        ml_report_summary += f"**--- Decision Tree Classifier ---**\n"
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(X_train, y_train)
        dt_predictions = dt_model.predict(X_test)
        
        ml_report_summary += f"Accuracy: **{accuracy_score(y_test, dt_predictions):.4f}**\n"
        ml_report_summary += f"Classification Report:\n{classification_report(y_test, dt_predictions, zero_division=0)}\n"
        
        # Confusion Matrix Plot for Decision Tree
        fig_dt_cm = plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, dt_predictions), annot=True, fmt='d', cmap='Blues', 
                    xticklabels=dt_model.classes_, yticklabels=dt_model.classes_)
        plt.title('Decision Tree Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plot_images['dt_confusion_matrix'] = get_image_base64(fig_dt_cm)

        # Feature Importance Plot for Decision Tree
        if hasattr(dt_model, 'feature_importances_') and len(X.columns) > 0:
            fig_dt_fi = plt.figure(figsize=(10, 6))
            importances_dt = pd.Series(dt_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            sns.barplot(x=importances_dt.values, y=importances_dt.index)
            plt.title('Decision Tree Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plot_images['dt_feature_importance'] = get_image_base64(fig_dt_fi)


        # --- Random Forest Classifier ---
        ml_report_summary += f"\n**--- Random Forest Classifier ---**\n"
        rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)

        ml_report_summary += f"Accuracy: **{accuracy_score(y_test, rf_predictions):.4f}**\n"
        ml_report_summary += f"Classification Report:\n{classification_report(y_test, rf_predictions, zero_division=0)}\n"

        # Confusion Matrix Plot for Random Forest
        fig_rf_cm = plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, rf_predictions), annot=True, fmt='d', cmap='Greens',
                    xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
        plt.title('Random Forest Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plot_images['rf_confusion_matrix'] = get_image_base64(fig_rf_cm)

        # Feature Importance Plot for Random Forest
        if hasattr(rf_model, 'feature_importances_') and len(X.columns) > 0:
            fig_rf_fi = plt.figure(figsize=(10, 6))
            importances_rf = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            sns.barplot(x=importances_rf.values, y=importances_rf.index)
            plt.title('Random Forest Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plot_images['rf_feature_importance'] = get_image_base64(fig_rf_fi)


        # --- Gradient Boosting Classifier ---
        ml_report_summary += f"\n**--- Gradient Boosting Classifier ---**\n"
        gb_model = GradientBoostingClassifier(random_state=42, n_estimators=100)
        gb_model.fit(X_train, y_train)
        gb_predictions = gb_model.predict(X_test)

        ml_report_summary += f"Accuracy: **{accuracy_score(y_test, gb_predictions):.4f}**\n"
        ml_report_summary += f"Classification Report:\n{classification_report(y_test, gb_predictions, zero_division=0)}\n"

        # Confusion Matrix Plot for Gradient Boosting
        fig_gb_cm = plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, gb_predictions), annot=True, fmt='d', cmap='Oranges',
                    xticklabels=gb_model.classes_, yticklabels=gb_model.classes_)
        plt.title('Gradient Boosting Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plot_images['gb_confusion_matrix'] = get_image_base64(fig_gb_cm)
        
        # Feature Importance Plot for Gradient Boosting
        if hasattr(gb_model, 'feature_importances_') and len(X.columns) > 0:
            fig_gb_fi = plt.figure(figsize=(10, 6))
            importances_gb = pd.Series(gb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            sns.barplot(x=importances_gb.values, y=importances_gb.index)
            plt.title('Gradient Boosting Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plot_images['gb_feature_importance'] = get_image_base64(fig_gb_fi)
            
    except FileNotFoundError:
        ml_report_summary = "Error: Data file not found. Please upload or generate data first."
        flash("Error: Data file not found.", "danger")
    except pd.errors.EmptyDataError:
        ml_report_summary = "Error: The uploaded file is empty. Please upload a file with data."
        flash("Error: The uploaded file is empty.", "danger")
    except KeyError as e:
        ml_report_summary = f"Error: Missing expected column in dataset: {e}. Please ensure your CSV has the correct format, especially the 'attack_type' target column."
        flash(f"Error: Missing expected column in dataset: {e}. Please check your CSV file.", "danger")
    except ValueError as e: # Catch the specific ValueError from train_test_split or other scikit-learn issues
        ml_report_summary = f"Error during ML analysis due to data distribution: {e}. This often means there are too few samples for some classes after processing. Please try a larger dataset, adjust the 'test_size', or group rare classes."
        flash(f"Data Error: {e}. Please try a larger dataset, or group rare attack types.", "danger")
        import traceback
        app.logger.error(f"ML ValueError (Train/Split/Data): {traceback.format_exc()}")
    except Exception as e:
        ml_report_summary = f"An unexpected error occurred during ML analysis: {e}"
        import traceback
        app.logger.error(f"ML Generic Error: {traceback.format_exc()}")
        flash(f"An unexpected error occurred during ML analysis: {e}. Check server logs for full details.", "danger")

    return render_template('ml_reports.html', ml_summary=ml_report_summary, plot_images=plot_images)


if __name__ == '__main__':
    # Set Matplotlib backend to 'Agg' which is non-interactive and suitable for web
    plt.switch_backend('Agg')
    app.run(debug=True) # Set debug=False in production for security and performance!