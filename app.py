"""
Student Performance Prediction — Flask Web Application
Run: python app.py
Open: http://127.0.0.1:5000
"""

import os, json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import secrets
import sqlite3
import logging

# Configure logging to see errors in Render logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
# Use SECRET_KEY env var in production, fallback to random for local dev
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(16))
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR: on Render this is /data (persistent disk), locally use 'data' folder
DATA_DIR   = os.environ.get('DATA_DIR', os.path.join(BASE_DIR, 'data'))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
DATA_PATH  = os.path.join(DATA_DIR, 'student_performance.csv')
SOURCE_CSV = os.path.join(BASE_DIR, 'student_performance.csv')  # bundled original

# Ensure DATA_DIR is writable. Fallback if /data fails (common on Free Tier)
try:
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
except (PermissionError, OSError):
    logger.warning(f"Could not create {DATA_DIR}, falling back to local 'data' folder")
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    os.makedirs(DATA_DIR, exist_ok=True)
    DATA_PATH = os.path.join(DATA_DIR, 'student_performance.csv')

# Copy CSV to DATA_DIR on first run (so /data has the data file)
if not os.path.exists(DATA_PATH) and os.path.exists(SOURCE_CSV):
    import shutil
    shutil.copy2(SOURCE_CSV, DATA_PATH)

# Load model once at startup
model = joblib.load(MODEL_PATH)

# Global df load logic with refresh capability
def load_current_df():
    try:
        if os.path.exists(DATA_PATH):
            new_df = pd.read_csv(DATA_PATH)
            new_df.fillna(new_df.median(numeric_only=True), inplace=True)
            return new_df
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
    
    # Fallback to empty with correct columns if file missing/corrupt
    return pd.DataFrame(columns=['study_hours','attendance','assignments_completed',
                               'previous_grade','participation','sleep_hours',
                               'internet_usage','final_score'])

df = load_current_df()

# Initialize predictions DB
def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(os.path.join(DATA_DIR, 'predictions.db'))
    cursor = conn.cursor()
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT,
            academic_year TEXT,
            study_hours REAL,
            attendance REAL,
            assignments_completed REAL,
            previous_grade REAL,
            participation REAL,
            sleep_hours REAL,
            internet_usage REAL,
            final_score REAL,
            grade TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Users table for admin login
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    # Check if admin exists, if not create default
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", ('admin', 'password'))
    
    # Student activity table for login tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS student_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT UNIQUE,
            login_time DATETIME,
            login_date TEXT,
            login_count INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()

    # ─── Teacher Database (Dedicated - Exact Requirements) ───
    conn_t = sqlite3.connect(os.path.join(DATA_DIR, 'teacher.db'))
    cursor_t = conn_t.cursor()
    
    # First, create the table if it doesn't even exist
    cursor_t.execute('''
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT UNIQUE,
            name TEXT,
            login_time DATETIME,
            daily_count INTEGER DEFAULT 0,
            last_login_date TEXT
        )
    ''')
    
    # Then ensure all columns exist (migration for teachers table)
    for col in [('roll_number', 'TEXT UNIQUE'), ('name', 'TEXT'), ('login_time', 'DATETIME'), ('daily_count', 'INTEGER DEFAULT 0'), ('last_login_date', 'TEXT')]:
        try:
            cursor_t.execute(f"ALTER TABLE teachers ADD COLUMN {col[0]} {col[1]}")
        except sqlite3.OperationalError:
            pass # Already exists or table is fresh
    
    conn_t.commit()
    conn_t.close()

init_db()

FEATURES = ['study_hours','attendance','assignments_completed',
            'previous_grade','participation','sleep_hours','internet_usage']

# ─── Helper ───────────────────────────────────────────────────────────────────
def get_grade(score):
    if score >= 90: return 'A+', 'Outstanding'
    if score >= 80: return 'A',  'Excellent'
    if score >= 70: return 'B',  'Good'
    if score >= 60: return 'C',  'Average'
    if score >= 50: return 'D',  'Below Average'
    return 'F', 'Needs Improvement'

def get_tips(data):
    tips = []
    if data['study_hours'] < 4:
        tips.append("📚 Increase study hours to at least 4–6 hours per day.")
    if data['attendance'] < 75:
        tips.append("🏫 Improve class attendance — aim for 80%+.")
    if data['assignments_completed'] < 7:
        tips.append("📝 Complete more assignments for better scores.")
    if data['sleep_hours'] < 6:
        tips.append("😴 Get adequate sleep (7–8 hrs) for better focus.")
    if data['internet_usage'] > 5:
        tips.append("📵 Reduce recreational internet usage during study time.")
    if not tips:
        tips.append("🎉 Keep up the excellent habits!")
    return tips

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.before_request
def require_login():
    # Global login check
    allowed_endpoints = ['login', 'static']
    if 'user' not in session and request.endpoint not in allowed_endpoints:
        return redirect(url_for('login'))


@app.route('/')
def index():
    global df
    df = load_current_df() # Refresh to see new data from other worker/process
    stats = {
        'total_students': len(df),
        'avg_score':      round(df['final_score'].mean() if not df.empty else 0, 1),
        'top_score':      round(df['final_score'].max() if not df.empty else 0, 1),
        'pass_rate':      round((df['final_score'] >= 50).mean() * 100 if not df.empty else 0, 1),
    }
    return render_template('index.html', stats=stats)




@app.route('/dashboard')
def dashboard():
    global df
    df = load_current_df() # Refresh
    
    # Summary metrics for the top cards
    total_records = len(df)
    avg_score = round(df['final_score'].mean(), 1) if total_records > 0 else 0
    pass_rate = round((df['final_score'] >= 50).mean() * 100, 1) if total_records > 0 else 0
    top_performers = int((df['final_score'] >= 90).sum()) if total_records > 0 else 0

    # Get the latest 100 results, assuming latest are at the bottom of the dataset
    recent_results = df.tail(100).iloc[::-1].to_dict(orient='records')

    # Grade distribution for a summary chart
    bins   = [0, 50, 60, 70, 80, 90, 101]
    labels = ['F (<50)', 'D (50-60)', 'C (60-70)', 'B (70-80)', 'A (80-90)', 'A+ (90+)']
    df['grade_bin'] = pd.cut(df['final_score'], bins=bins, labels=labels, right=False)
    grade_dist = df['grade_bin'].value_counts().sort_index().to_dict()

    # Model metrics (hard-coded from training for display)
    metrics = {
        'Linear Regression': {'R2': 0.8957, 'MAE': 3.31, 'MSE': 18.07},
        'Decision Tree':     {'R2': 0.6959, 'MAE': 5.64, 'MSE': 52.70},
        'Random Forest':     {'R2': 0.8592, 'MAE': 3.88, 'MSE': 24.39},
    }

    # Data for live interactive charts
    study_vs_score = {'x': df['study_hours'].tolist(), 'y': df['final_score'].tolist()}
    
    att_bins = [0, 50, 60, 70, 80, 90, 101]
    att_labels = ['<50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    df['att_bin'] = pd.cut(df['attendance'], bins=att_bins, labels=att_labels, right=False)
    # Using observed=False to silence Pandas warning for categorical groupby
    att_vs_score = df.groupby('att_bin', observed=False)['final_score'].mean().fillna(0).to_dict()
    
    prev_vs_score = {'x': df['previous_grade'].tolist(), 'y': df['final_score'].tolist()}
    
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=30, random_state=42)
    rf.fit(df[FEATURES], df['final_score'])
    feat_imp_data = {'labels': FEATURES, 'values': rf.feature_importances_.tolist()}
    
    corr_cols = FEATURES + ['final_score']
    heatmap_data = {
        'z': df[corr_cols].corr().values.tolist(),
        'x': corr_cols,
        'y': corr_cols
    }
    
    chart_data = {
        'study_vs_score': study_vs_score,
        'att_vs_score': att_vs_score,
        'prev_vs_score': prev_vs_score,
        'feat_imp': feat_imp_data,
        'heatmap': heatmap_data
    }

    summary = {
        'total': total_records,
        'avg_score': avg_score,
        'pass_rate': pass_rate,
        'top_performers': top_performers
    }

    return render_template('dashboard.html', summary=summary,
                           results=recent_results, grade_dist=json.dumps(grade_dist),
                           metrics=metrics, chart_data=json.dumps(chart_data))


# ─── Auth Routes ──────────────────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('index'))
        
    error = None
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            password = request.form.get('password')
            
            logger.info(f"Login attempt for user: {username}")
            
            conn = sqlite3.connect(os.path.join(DATA_DIR, 'predictions.db'))
            cursor = conn.cursor()
            
            # 1. Check for Admin
            if username == 'admin' and (password == 'password' or password == 'Admin@123'):
                session['user'] = 'admin'
                session['role'] = 'admin'
                conn.close()
                return redirect(url_for('index'))
                
            # 2. Check for Teacher Role
            conn_t = sqlite3.connect(os.path.join(DATA_DIR, 'teacher.db'))
            cursor_t = conn_t.cursor()
            cursor_t.execute("SELECT roll_number, name FROM teachers WHERE UPPER(roll_number) = UPPER(?)", (username,))
            t_row = cursor_t.fetchone()
            
            if t_row:
                verified_roll = t_row[0]
                teacher_name = t_row[1]
                if password == verified_roll[1:5]:
                    session['role'] = 'teacher'
                    session['user'] = verified_roll
                    session['name'] = teacher_name
                    session['predict_mode'] = False
                    
                    from datetime import datetime
                    today_str = datetime.now().strftime('%Y-%m-%d')
                    
                    cursor_t.execute("SELECT daily_count, last_login_date FROM teachers WHERE roll_number = ?", (verified_roll,))
                    row_t = cursor_t.fetchone()
                    
                    count = 1
                    if row_t and row_t[1] == today_str:
                        count = (row_t[0] or 0) + 1
                    
                    cursor_t.execute('''
                        UPDATE teachers 
                        SET login_time = CURRENT_TIMESTAMP, 
                            daily_count = ?, 
                            last_login_date = ? 
                        WHERE roll_number = ?
                    ''', (count, today_str, verified_roll))
                    
                    conn_t.commit()
                    conn_t.close()
                    conn.close()
                    return redirect(url_for('index'))
                else:
                    error = "Invalid Teacher Password"
            conn_t.close()
            
            # 3. Check for Student Role
            if not t_row and not error:
                cursor.execute("SELECT roll_number FROM predictions WHERE UPPER(roll_number) = UPPER(?)", (username,))
                s_row = cursor.fetchone()
                if s_row:
                    roll_num = s_row[0]
                    if password == roll_num[-4:]:
                        session['user'] = roll_num
                        session['role'] = 'student'
                        
                        from datetime import datetime
                        now = datetime.now()
                        login_time = now.strftime('%Y-%m-%d %H:%M:%S')
                        login_date = now.strftime('%Y-%m-%d')
                        
                        cursor.execute("SELECT login_count, login_date FROM student_activity WHERE roll_number = ?", (roll_num,))
                        s_row_act = cursor.fetchone()
                        
                        count = 1
                        if s_row_act:
                            if s_row_act[1] == login_date:
                                count = (s_row_act[0] or 0) + 1
                            
                            cursor.execute('''
                                UPDATE student_activity 
                                SET login_time = ?, login_date = ?, login_count = ?
                                WHERE roll_number = ?
                            ''', (login_time, login_date, count, roll_num))
                        else:
                            cursor.execute('''
                                INSERT INTO student_activity (roll_number, login_time, login_date, login_count)
                                VALUES (?, ?, ?, ?)
                            ''', (roll_num, login_time, login_date, count))
                        
                        conn.commit()
                        conn.close()
                        return redirect(url_for('index'))
                    else:
                        error = "Invalid Student Password"
                else:
                    error = "Roll Number Not Found"
                
            conn.close()
        except Exception as e:
            logger.error(f"Login Error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Login Error: {str(e)}", 500
            
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('role', None)
    return redirect(url_for('login'))



@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/department', methods=['GET'])
def department():
    return render_template('department.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/view_csv', methods=['GET'])
def view_csv():
    global df
    df = load_current_df() # Refresh
    filter_type = request.args.get('filter')
    
    display_df = df.copy()
    title_suffix = ""
    
    if filter_type == 'top':
        max_score = df['final_score'].max()
        display_df = df[df['final_score'] == max_score]
        title_suffix = " - Top Scorers"
    elif filter_type == 'pass':
        display_df = df[df['final_score'] >= 50]
        title_suffix = " - Passed Students"
    elif filter_type == 'average':
        avg = df['final_score'].mean()
        display_df = df[(df['final_score'] >= avg - 5) & (df['final_score'] <= avg + 5)]
        title_suffix = " - Average Scorers"
        
    return render_template('view_csv.html', 
                           table=display_df.to_html(classes='table table-dark table-striped table-hover', index=False),
                           suffix=title_suffix)


@app.route('/view_database', methods=['GET'])
def view_database():
    try:
        conn = sqlite3.connect(os.path.join(DATA_DIR, 'predictions.db'))
        db_df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
        conn.close()
        
        # Format the table similar to view_csv
        table_html = db_df.to_html(classes='table table-dark table-striped table-hover', index=False)
        return render_template('view_database.html', table=table_html)
    except Exception as e:
        return f"Error accessing database: {e}", 500


@app.route('/view_teachers', methods=['GET'])
def view_teachers():
    if session.get('role') != 'admin':
        return redirect(url_for('index'))
    try:
        conn = sqlite3.connect(os.path.join(DATA_DIR, 'teacher.db'))
        # Using row_factory to get dictionaries/easier access
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM teachers ORDER BY id DESC")
        teachers = cursor.fetchall()
        conn.close()
        return render_template('view_teachers.html', teachers=teachers)
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/view_student_activity', methods=['GET'])
def view_student_activity():
    if session.get('role') != 'admin':
        return redirect(url_for('index'))
    try:
        conn = sqlite3.connect(os.path.join(DATA_DIR, 'predictions.db'))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM student_activity ORDER BY login_time DESC")
        students = cursor.fetchall()
        conn.close()
        return render_template('view_students.html', students=students)
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/add_teacher', methods=['POST'])
def add_teacher():
    if session.get('role') != 'admin':
        return redirect(url_for('index'))
    roll_number = request.form.get('roll_number')
    name = request.form.get('name', 'N/A')
    if roll_number:
        try:
            conn = sqlite3.connect(os.path.join(DATA_DIR, 'teacher.db'))
            cursor = conn.cursor()
            cursor.execute("INSERT INTO teachers (roll_number, name) VALUES (?, ?)", (roll_number.upper(), name))
            conn.commit()
            conn.close()
        except Exception as e:
            return f"Error adding teacher: {str(e)}", 500
    return redirect(url_for('view_teachers'))

@app.route('/delete_teacher/<int:id>')
def delete_teacher(id):
    if session.get('role') != 'admin':
        return redirect(url_for('index'))
    try:
        conn = sqlite3.connect(os.path.join(DATA_DIR, 'teacher.db'))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM teachers WHERE id = ?", (id,))
        conn.commit()
        conn.close()
    except Exception as e:
        return f"Error deleting teacher: {str(e)}", 500
    return redirect(url_for('view_teachers'))


@app.route('/predict', methods=['GET'])
def predict():
    if session.get('role') == 'student':
        return redirect(url_for('student_predict'))
    # Teachers need active predict_mode to see this page
    if session.get('role') == 'teacher' and not session.get('predict_mode'):
        return redirect(url_for('index'))
    return render_template('predict.html')


@app.route('/start_predict')
def start_predict():
    role = session.get('role')
    if role in ['admin', 'teacher']:
        session['predict_mode'] = True
        return redirect(url_for('predict'))
    elif role == 'student':
        return redirect(url_for('student_predict'))
    return redirect(url_for('index'))

@app.route('/student_predict', methods=['GET', 'POST'])
def student_predict():
    if request.method == 'POST':
        roll_number = session.get('user')
        academic_year = request.form.get('academic_year')
        
        try:
            conn = sqlite3.connect(os.path.join(DATA_DIR, 'predictions.db'))
            cursor = conn.cursor()
            # Fetch the most recent prediction for this student and semester
            cursor.execute('''
                SELECT * FROM predictions 
                WHERE UPPER(roll_number) = UPPER(?) AND academic_year = ?
                ORDER BY id DESC LIMIT 1
            ''', (roll_number, academic_year))
            row = cursor.fetchone()
            
            if row:
                # Column mapping based on schema
                # ID: 0, Name: id, Type: INTEGER
                # ID: 1, Name: roll_number, Type: TEXT
                # ID: 2, Name: academic_year, Type: TEXT
                # ...
                data = {
                    'study_hours': row[3],
                    'attendance': row[4],
                    'assignments_completed': row[5],
                    'previous_grade': row[6],
                    'participation': row[7],
                    'sleep_hours': row[8],
                    'internet_usage': row[9],
                }
                score = row[10]
                grade = row[11]
                label = get_grade(score)[1]
                tips = get_tips(data)
                
                # Fetch global stats for template compatibility
                avg_score_global = round(df['final_score'].mean(), 1)
                pass_rate_global = round((df['final_score'] >= 50).mean() * 100, 1)
                
                # Historical chart data
                historical_data = {}
                cursor.execute('SELECT academic_year, final_score FROM predictions WHERE UPPER(roll_number) = UPPER(?) ORDER BY id ASC', (roll_number,))
                h_rows = cursor.fetchall()
                for h in h_rows:
                    if h[0] and h[0] != 'N/A':
                        historical_data[h[0]] = float(h[1])
                
                conn.close()
                return render_template('result.html',
                                       score=round(score, 1),
                                       grade=grade,
                                       label=label,
                                       tips=tips,
                                       percentile=0, # Placeholder
                                       inputs=data,
                                       roll_number=roll_number,
                                       academic_year=academic_year,
                                       global_stats={'total': len(df), 'avg': avg_score_global, 'pass_rate': pass_rate_global},
                                       historical_data=json.dumps(historical_data))
            else:
                conn.close()
                return render_template('student_predict.html', error=f"No results found for {academic_year}. Please contact admin if you believe this is an error.")
        except Exception as e:
            return render_template('student_predict.html', error=f"Database error: {e}")

    return render_template('student_predict.html', roll_number=session.get('user'))


@app.route('/add_manual', methods=['GET', 'POST'])
def add_manual():
    # Teachers need active predict_mode to see this page
    if session.get('role') == 'teacher' and not session.get('predict_mode'):
        return redirect(url_for('index'))
    global df
    if request.method == 'POST':
        try:
            roll_number = request.form.get('roll_number', 'N/A')
            academic_year = request.form.get('academic_year', 'N/A')
            score = float(request.form['final_score'])
            grade, label = get_grade(score)
            
            data = {
                'study_hours':           float(request.form['study_hours']),
                'attendance':            float(request.form['attendance']),
                'assignments_completed': float(request.form['assignments_completed']),
                'previous_grade':        float(request.form.get('previous_grade') or df['previous_grade'].median()),
                'participation':         float(request.form['participation']),
                'sleep_hours':           float(request.form['sleep_hours']),
                'internet_usage':        float(request.form.get('internet_usage') or df['internet_usage'].median()),
            }

            # Append to dataset
            new_entry = data.copy()
            new_entry['final_score'] = score
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            try:
                save_df = df.drop(columns=['grade_bin', 'att_bin'], errors='ignore')
                save_df.to_csv(DATA_PATH, index=False)
                logger.info(f"CSV updated successfully at {DATA_PATH}")
            except Exception as e:
                logger.error(f"Failed to save CSV: {e}")

            # Store in Database
            try:
                conn = sqlite3.connect(os.path.join(DATA_DIR, 'predictions.db'))
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO predictions (
                        roll_number, academic_year, study_hours, attendance, 
                        assignments_completed, previous_grade, participation, 
                        sleep_hours, internet_usage, final_score, grade
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    roll_number, academic_year, data['study_hours'], data['attendance'],
                    data['assignments_completed'], data['previous_grade'], data['participation'],
                    data['sleep_hours'], data['internet_usage'], score, grade
                ))
                conn.commit()
                conn.close()
                return render_template('add_manual.html', success=f"Data for {roll_number} ({academic_year}) saved successfully!")
            except Exception as e:
                return render_template('add_manual.html', error=f"Database insertion error: {e}")

        except Exception as e:
            return render_template('add_manual.html', error=f"Error: {str(e)}")

    return render_template('add_manual.html')


@app.route('/api/predict_score', methods=['POST'])
def api_predict_score():
    try:
        data = request.json
        features = {
            'study_hours': float(data.get('study_hours') or 0),
            'attendance': float(data.get('attendance') or 0),
            'assignments_completed': float(data.get('assignments_completed') or 0),
            'previous_grade': float(data.get('previous_grade') or df['previous_grade'].median()),
            'participation': float(data.get('participation') or 0),
            'sleep_hours': float(data.get('sleep_hours') or 0),
            'internet_usage': float(data.get('internet_usage') or df['internet_usage'].median()),
        }
        X = pd.DataFrame([features], columns=FEATURES)
        score = float(np.clip(model.predict(X)[0], 0, 100))
        return jsonify({'score': round(score, 1)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/<roll_number>', methods=['GET'])
def api_history(roll_number):
    historical_data = {}
    try:
        conn = sqlite3.connect(os.path.join(DATA_DIR, 'predictions.db'))
        cursor = conn.cursor()
        cursor.execute('''
            SELECT academic_year, final_score
            FROM predictions
            WHERE roll_number = ?
            ORDER BY id ASC
        ''', (roll_number.upper(),))
        rows = cursor.fetchall()
        for r in rows:
            if r[0] and r[0] != 'N/A':
                historical_data[r[0]] = float(r[1])
        conn.close()
    except Exception as e:
        print(f"Error fetching API historical data: {e}")
    return jsonify(historical_data)


@app.route('/result', methods=['POST'])
def result():
    global df
    try:
        previous_prediction = session.get('previous_prediction')
        roll_number = request.form.get('roll_number', 'N/A')
        academic_year = request.form.get('academic_year', 'N/A')
        data = {
            'study_hours':           float(request.form['study_hours']),
            'attendance':            float(request.form['attendance']),
            'assignments_completed': float(request.form['assignments_completed']),
            'previous_grade':        float(request.form.get('previous_grade') or df['previous_grade'].median()),
            'participation':         float(request.form['participation']),
            'sleep_hours':           float(request.form['sleep_hours']),
            'internet_usage':        float(request.form.get('internet_usage') or df['internet_usage'].median()),
        }

        X = pd.DataFrame([data], columns=FEATURES)
        score = float(np.clip(model.predict(X)[0], 0, 100))
        grade, label = get_grade(score)
        tips = get_tips(data)

        # Percentile rank in dataset
        percentile = round((df['final_score'] < score).mean() * 100, 1)

        # Fetch global stats for comparison
        total_records = len(df)
        avg_score_global = round(df['final_score'].mean(), 1)
        pass_rate_global = round((df['final_score'] >= 50).mean() * 100, 1)
        
        # Append prediction live to dataset to reflect in global stats
        new_entry = data.copy()
        new_entry['final_score'] = score
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        try:
            save_df = df.drop(columns=['grade_bin', 'att_bin'], errors='ignore')
            save_df.to_csv(DATA_PATH, index=False)
            logger.info(f"CSV updated successfully at {DATA_PATH}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

        # Store in SQLite Database
        try:
            conn = sqlite3.connect(os.path.join(DATA_DIR, 'predictions.db'))
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (
                    roll_number, academic_year, study_hours, attendance, 
                    assignments_completed, previous_grade, participation, 
                    sleep_hours, internet_usage, final_score, grade
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                roll_number, academic_year, data['study_hours'], data['attendance'],
                data['assignments_completed'], data['previous_grade'], data['participation'],
                data['sleep_hours'], data['internet_usage'], score, grade
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database insertion error: {e}")

        session['previous_prediction'] = {
            'score': round(score, 1),
            'grade': grade,
            'label': label,
            'inputs': data
        }

        # Fetch historical data for chart
        historical_data = {}
        if roll_number and roll_number != 'N/A':
            try:
                conn = sqlite3.connect(os.path.join(DATA_DIR, 'predictions.db'))
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT academic_year, final_score
                    FROM predictions
                    WHERE roll_number = ?
                    ORDER BY id ASC
                ''', (roll_number,))
                rows = cursor.fetchall()
                for r in rows:
                    if r[0] and r[0] != 'N/A':
                        historical_data[r[0]] = float(r[1])
                conn.close()
            except Exception as e:
                print(f"Error fetching historical data: {e}")

        return render_template('result.html',
                               score=round(score, 1),
                               grade=grade,
                               label=label,
                               tips=tips,
                               percentile=percentile,
                               inputs=data,
                               roll_number=roll_number,
                               academic_year=academic_year,
                               previous_data=previous_prediction,
                               global_stats={
                                   'total': total_records,
                                   'avg': avg_score_global,
                                   'pass_rate': pass_rate_global
                               },
                               historical_data=json.dumps(historical_data))
    except Exception as e:
        return render_template('predict.html', error=f"Error: {str(e)}")


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n  Student Performance Prediction Web App")
    print("  Open → http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
