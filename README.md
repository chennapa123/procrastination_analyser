# 🧠 Procrastination Score Predictor

A Streamlit web app that predicts your procrastination score (0–5) using a Random Forest model trained on your daily productivity habits.

## 📁 Folder Structure

```
procrastination_app/
├── app.py                  # Main Streamlit application
├── dataset-uncleaned.csv   # Raw dataset (model trains on this at startup)
├── requirements.txt        # Python dependencies
└── README.md
```

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 Features

- **Interactive sliders** for all habit inputs (sleep, screen time, stress, etc.)
- **Instant prediction** using a trained Random Forest Regressor
- **Color-coded score card** (green → red)
- **Personalised tips** based on your specific inputs
- **Task completion rate** insight

## 📊 Input Features

| Feature | Description |
|---|---|
| Age | User age (16–40) |
| Sleep Hours | Hours of sleep per night |
| Screen Time (Minutes) | Daily screen usage in minutes |
| Tasks Planned | Number of tasks planned for the day |
| Tasks Completed | Tasks actually finished |
| Focus Hours | Hours of deep focused work |
| Stress Level | Self-rated stress (1–10) |
| Motivation Level | Self-rated motivation (1–10) |
| Task Interest | Engagement with tasks (1–10) |
| Self Control | Ability to resist distractions (1–10) |
| Routine Consistency | Whether your routine is consistent (Yes/No) |
| Peak Productivity Time | Morning / Afternoon / Evening |

## 📈 Model

- **Algorithm**: Random Forest Regressor (100 trees)
- **Target**: `procrastination_score` (0 = no procrastination, 5 = high procrastination)
- **Training split**: 80/20
- The model is trained fresh each time the app starts (cached in session)
