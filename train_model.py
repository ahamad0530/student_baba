"""
Student Performance Prediction - Model Training Script
Generates synthetic dataset, performs EDA, trains ML models, and saves the best model.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
IMG_DIR   = os.path.join(BASE_DIR, 'static', 'images')
DATA_PATH = os.path.join(BASE_DIR, 'student_performance.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
os.makedirs(IMG_DIR, exist_ok=True)

# ─── 1. Generate Realistic Synthetic Dataset ──────────────────────────────────
def generate_dataset(n=1200, seed=42):
    rng = np.random.default_rng(seed)

    study_hours         = rng.uniform(1, 10, n)
    attendance          = rng.uniform(50, 100, n)
    assignments_done    = rng.integers(0, 11, n).astype(float)
    previous_grade      = rng.uniform(40, 100, n)
    participation       = rng.uniform(1, 10, n)
    sleep_hours         = rng.uniform(4, 10, n)
    internet_usage      = rng.uniform(1, 8, n)

    # Score formula with realistic weights + noise
    noise = rng.normal(0, 4, n)
    final_score = (
        3.5  * study_hours +
        0.25 * attendance +
        1.8  * assignments_done +
        0.30 * previous_grade +
        1.2  * participation +
        0.9  * sleep_hours -
        0.8  * internet_usage +
        noise
    )
    # Clip to [0, 100]
    final_score = np.clip(final_score, 0, 100)

    df = pd.DataFrame({
        'study_hours':           np.round(study_hours, 2),
        'attendance':            np.round(attendance, 2),
        'assignments_completed': assignments_done,
        'previous_grade':        np.round(previous_grade, 2),
        'participation':         np.round(participation, 2),
        'sleep_hours':           np.round(sleep_hours, 2),
        'internet_usage':        np.round(internet_usage, 2),
        'final_score':           np.round(final_score, 2),
    })

    # Introduce ~5% missing values for realism
    for col in ['study_hours', 'sleep_hours', 'internet_usage']:
        mask = rng.random(n) < 0.05
        df.loc[mask, col] = np.nan

    df.to_csv(DATA_PATH, index=False)
    print(f"[✓] Dataset saved → {DATA_PATH}  ({n} rows)")
    return df

# ─── 2. Preprocess ────────────────────────────────────────────────────────────
def preprocess(df):
    print(f"\n[INFO] Missing values before cleaning:\n{df.isnull().sum()}")
    df = df.fillna(df.median(numeric_only=True))
    print("[✓] Missing values filled with column medians.")
    return df

# ─── 3. EDA Plots ─────────────────────────────────────────────────────────────
PALETTE = "#4F46E5"   # indigo accent
BG      = "#0F0F1A"
FG      = "#E2E8F0"

def _style():
    plt.rcParams.update({
        'figure.facecolor': BG,
        'axes.facecolor':   BG,
        'axes.edgecolor':   '#2D2D44',
        'axes.labelcolor':  FG,
        'xtick.color':      FG,
        'ytick.color':      FG,
        'text.color':       FG,
        'grid.color':       '#2D2D44',
        'grid.linestyle':   '--',
        'grid.alpha':       0.5,
        'font.family':      'DejaVu Sans',
    })

def plot_study_vs_score(df):
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df['study_hours'], df['final_score'], alpha=0.45, color=PALETTE, edgecolors='none', s=18)
    m, b = np.polyfit(df['study_hours'].dropna(), df['final_score'], 1)
    xs = np.linspace(df['study_hours'].min(), df['study_hours'].max(), 100)
    ax.plot(xs, m*xs+b, color='#F59E0B', lw=2, label='Trend')
    ax.set_xlabel('Study Hours / Day')
    ax.set_ylabel('Final Score')
    ax.set_title('Study Hours vs Final Score', fontsize=14, fontweight='bold', color=FG)
    ax.legend(facecolor='#1E1E30', edgecolor='none', labelcolor=FG)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'study_vs_score.png'), dpi=120)
    plt.close()
    print("[✓] study_vs_score.png")

def plot_attendance_vs_score(df):
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = [50, 60, 70, 80, 90, 100]
    labels = ['50-60', '60-70', '70-80', '80-90', '90-100']
    df2 = df.copy()
    df2['att_bin'] = pd.cut(df2['attendance'], bins=bins, labels=labels)
    means = df2.groupby('att_bin', observed=True)['final_score'].mean()
    colors = ['#6366F1','#818CF8','#A5B4FC','#C7D2FE','#E0E7FF']
    bars = ax.bar(means.index, means.values, color=colors, edgecolor='none', width=0.6)
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{val:.1f}', ha='center', va='bottom', fontsize=10, color=FG)
    ax.set_xlabel('Attendance Range (%)')
    ax.set_ylabel('Average Final Score')
    ax.set_title('Attendance vs Average Performance', fontsize=14, fontweight='bold', color=FG)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'attendance_vs_score.png'), dpi=120)
    plt.close()
    print("[✓] attendance_vs_score.png")

def plot_prev_grade_vs_score(df):
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hexbin(df['previous_grade'], df['final_score'], gridsize=30, cmap='Blues', linewidths=0)
    ax.set_xlabel('Previous Grade')
    ax.set_ylabel('Final Score')
    ax.set_title('Previous Grade vs Final Score', fontsize=14, fontweight='bold', color=FG)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'prev_grade_vs_score.png'), dpi=120)
    plt.close()
    print("[✓] prev_grade_vs_score.png")

def plot_heatmap(df):
    _style()
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                linewidths=0.5, linecolor='#1E1E30', ax=ax,
                annot_kws={'size': 9}, vmin=-1, vmax=1,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', color=FG)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'heatmap.png'), dpi=120)
    plt.close()
    print("[✓] heatmap.png")

def plot_feature_importance(model, features):
    _style()
    importances = model.feature_importances_
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(features)))
    ax.barh([features[i] for i in idx], importances[idx], color=colors[idx], edgecolor='none')
    ax.set_xlabel('Importance Score')
    ax.set_title('Random Forest — Feature Importance', fontsize=14, fontweight='bold', color=FG)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'feature_importance.png'), dpi=120)
    plt.close()
    print("[✓] feature_importance.png")

# ─── 4. Train Models ──────────────────────────────────────────────────────────
def train_models(df):
    features = ['study_hours','attendance','assignments_completed',
                'previous_grade','participation','sleep_hours','internet_usage']
    X = df[features]
    y = df['final_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree':     DecisionTreeRegressor(max_depth=6, random_state=42),
        'Random Forest':     RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42),
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        r2  = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        results[name] = {'model': m, 'R2': r2, 'MAE': mae, 'MSE': mse}
        print(f"  [{name}]  R²={r2:.4f}  MAE={mae:.4f}  MSE={mse:.4f}")

    # Pick best by R²
    best_name = max(results, key=lambda k: results[k]['R2'])
    best_model = results[best_name]['model']
    print(f"\n[✓] Best model: {best_name}  (R²={results[best_name]['R2']:.4f})")

    # Save
    joblib.dump(best_model, MODEL_PATH)
    print(f"[✓] Model saved → {MODEL_PATH}")

    # Feature importance plot (only for tree-based)
    plot_feature_importance(results['Random Forest']['model'], features)

    # Save metrics for dashboard
    metrics = {k: {mk: mv for mk, mv in v.items() if mk != 'model'} for k, v in results.items()}
    return metrics

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  Student Performance Prediction — Training Pipeline")
    print("=" * 55)

    df_raw = generate_dataset()
    df     = preprocess(df_raw)

    print("\n[STEP] Generating EDA plots …")
    plot_study_vs_score(df)
    plot_attendance_vs_score(df)
    plot_prev_grade_vs_score(df)
    plot_heatmap(df)

    print("\n[STEP] Training models …")
    metrics = train_models(df)

    print("\n[✓] Training complete. Run  python app.py  to start the web app.")
