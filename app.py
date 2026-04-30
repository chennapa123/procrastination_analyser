import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

st.set_page_config(
    page_title="Procrastination Score Predictor",
    page_icon="🧠",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .block-container {
        background: white;
        border-radius: 20px;
        padding: 2rem 3rem;
        margin-top: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
    }
    h1 { color: #4a0e8f; font-weight: 800; }
    h3 { color: #5a4f7c; }
    .score-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1.5rem 0;
    }
    .score-number { font-size: 4rem; font-weight: 900; }
    .score-label { font-size: 1.2rem; opacity: 0.9; }
    .tip-box {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }
    .stSlider > div > div { color: #4a0e8f; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Model Training (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on dataset…")
def load_model():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataset-uncleaned.csv"))

    # Fill missing values
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    df.drop_duplicates(inplace=True)

    # Encode categoricals and keep per-column encoders for inference
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])

    X = df.drop('procrastination_score', axis=1)
    y = df['procrastination_score']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns.tolist(), encoders


model, feature_cols, encoders = load_model()


# ── Helpers ───────────────────────────────────────────────────────────────────
def score_label(score):
    if score <= 1:
        return "🟢 Very Low", "You're highly productive and self-disciplined!"
    elif score <= 2:
        return "🔵 Low", "Good focus habits — minor improvements possible."
    elif score <= 3:
        return "🟡 Moderate", "You procrastinate occasionally. Room to grow!"
    elif score <= 4:
        return "🟠 High", "Procrastination is affecting your productivity."
    else:
        return "🔴 Very High", "Significant procrastination patterns detected."

def get_tips(score, inputs):
    tips = []
    if inputs['sleep_hours'] < 7:
        tips.append("💤 Aim for 7–9 hours of sleep to boost focus and reduce procrastination.")
    if inputs['screen_time_minutes'] > 240:
        tips.append("📵 Reduce screen time. Excessive use drains motivation and self-control.")
    if inputs['stress_level'] > 6:
        tips.append("🧘 High stress is a procrastination trigger. Try mindfulness or exercise.")
    if inputs['motivation_level'] < 4:
        tips.append("🎯 Set small, achievable daily goals to rebuild motivation momentum.")
    if inputs['self_control'] < 4:
        tips.append("🛡️ Practice time-blocking: schedule focused work intervals (e.g., Pomodoro).")
    if inputs['routine_consistency'] == "No":
        tips.append("📅 Build a consistent daily routine — consistency reduces decision fatigue.")
    if not tips:
        tips.append("✅ Keep up your great habits! Stay consistent and keep challenging yourself.")
    return tips


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Procrastination Score Predictor")
st.markdown("Fill in your daily habits below and get your **predicted procrastination score** (0 = best, 5 = worst).")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### � Sleep & Screen")
    age = st.slider("Age", 16, 40, 24)
    sleep_hours = st.slider("Sleep Hours per Night", 3.0, 12.0, 7.0, 0.5)
    screen_time_minutes = st.slider("Daily Screen Time (minutes)", 30, 720, 240, 5)

    st.markdown("### ✅ Task Management")
    tasks_planned = st.number_input("Tasks Planned Today", 1, 20, 5)
    tasks_completed = st.number_input("Tasks Completed Today", 0, 20, 3)
    focus_hours = st.slider("Deep Focus Hours", 0.0, 10.0, 3.0, 0.5)

with col2:
    st.markdown("### 😓 Stress & Motivation")
    stress_level = st.slider("Stress Level (1 = calm, 10 = very stressed)", 1, 10, 5)
    motivation_level = st.slider("Motivation Level (1 = low, 10 = high)", 1, 10, 5)

    st.markdown("### 🎯 Behavior & Routine")
    task_interest = st.slider("Task Interest / Engagement (1–10)", 1.0, 10.0, 5.0, 0.5)
    self_control = st.slider("Self-Control (1 = low, 10 = high)", 1.0, 10.0, 5.0, 0.5)
    routine_consistency = st.selectbox("Routine Consistency", options=["Yes", "No"])
    peak_productivity_time = st.selectbox(
        "Peak Productivity Time",
        options=["Morning", "Afternoon", "Evening"],
        format_func=lambda x: {"Morning": "🌅 Morning", "Afternoon": "🌞 Afternoon", "Evening": "🌙 Evening"}[x]
    )

st.divider()

if st.button("🔍 Predict My Procrastination Score", use_container_width=True, type="primary"):
    input_dict = {
        'age': age,
        'sleep_hours': sleep_hours,
        'screen_time_minutes': screen_time_minutes,
        'tasks_planned': tasks_planned,
        'tasks_completed': tasks_completed,
        'focus_hours': focus_hours,
        'stress_level': stress_level,
        'motivation_level': motivation_level,
        'task_interest': task_interest,
        'self_control': self_control,
        'routine_consistency': routine_consistency,
        'peak_productivity_time': peak_productivity_time,
    }

    input_df = pd.DataFrame([input_dict])[feature_cols]
    for col, enc in encoders.items():
        input_df[col] = enc.transform(input_df[col].astype(str))
    raw_score = model.predict(input_df)[0]
    score = round(float(np.clip(raw_score, 0, 5)), 2)

    label, summary = score_label(score)
    tips = get_tips(score, input_dict)

    st.markdown(f"""
    <div class="score-card">
        <div class="score-label">Your Procrastination Score</div>
        <div class="score-number">{score} / 5</div>
        <div class="score-label">{label}</div>
        <div style="margin-top:0.5rem; font-size:1rem; opacity:0.85;">{summary}</div>
    </div>
    """, unsafe_allow_html=True)

    # Progress bar
    progress = score / 5
    bar_color = "#22c55e" if score <= 1 else "#3b82f6" if score <= 2 else "#eab308" if score <= 3 else "#f97316" if score <= 4 else "#ef4444"
    st.markdown(f"""
    <div style="background:#e5e7eb; border-radius:50px; height:18px; overflow:hidden; margin-bottom:1.5rem;">
        <div style="background:{bar_color}; width:{progress*100:.1f}%; height:100%; border-radius:50px; transition:width 0.5s;"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 💡 Personalised Tips")
    for tip in tips:
        st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

    # Completion rate insight
    comp_rate = tasks_completed / (tasks_planned + 1)
    st.markdown(f"""
    <div class="tip-box" style="margin-top:1rem;">
    📊 <strong>Task Completion Rate:</strong> {comp_rate*100:.0f}%
    {"— Great job finishing your tasks!" if comp_rate >= 0.7 else "— Try to complete more of your planned tasks each day."}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Model: Random Forest Regressor · Trained on productivity & habit dataset · Score range: 0 (no procrastination) → 5 (high procrastination)")
