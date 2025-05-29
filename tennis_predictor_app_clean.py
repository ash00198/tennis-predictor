
import streamlit as st
import joblib
import pandas as pd

# Load the model and precomputed data
model = joblib.load("random_forest_tennis_model_small.pkl")
clay_win_pct = joblib.load("clay_win_pct.pkl")
h2h_wins = joblib.load("h2h_wins.pkl")

# Function to compute match features
def compute_features(p1, p2, r1, r2, o1, o2):
    clay1 = clay_win_pct.get(p1, 0.5)
    clay2 = clay_win_pct.get(p2, 0.5)
    clay_diff = clay1 - clay2
    h2h_diff = h2h_wins.get(p1, {}).get(p2, 0) - h2h_wins.get(p2, {}).get(p1, 0)
    rank_diff = r1 - r2
    odds_diff = o2 - o1

    return pd.DataFrame([{
        'Clay_Win_%_P1': clay1,
        'Clay_Win_%_P2': clay2,
        'Clay_Win_%_Diff': clay_diff,
        'H2H_Diff': h2h_diff,
        'Rank_Diff': rank_diff,
        'Odds_Diff': odds_diff
    }])

# UI
st.title("🎾 Tennis Match Predictor (Clay Court)")

tabs = st.tabs(["Single Match Prediction", "Batch Prediction"])

with tabs[0]:
    p1 = st.text_input("Player 1")
    p2 = st.text_input("Player 2")
    r1 = st.number_input("Rank Player 1", value=50)
    r2 = st.number_input("Rank Player 2", value=50)
    o1 = st.number_input("Odds Player 1", value=1.5)
    o2 = st.number_input("Odds Player 2", value=2.5)

    if st.button("Predict Winner"):
        features = compute_features(p1, p2, r1, r2, o1, o2)
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1 if pred else 0]
        winner = p1 if pred else p2
        st.success(f"Prediction: {winner} wins with {round(prob*100, 2)}% confidence.")

with tabs[1]:
    st.markdown("Upload a CSV with columns: `Player_1`, `Player_2`, `Rank_1`, `Rank_2`, `Odd_1`, `Odd_2`")
    uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        results = []
        for _, row in df.iterrows():
            features = compute_features(row['Player_1'], row['Player_2'], row['Rank_1'], row['Rank_2'], row['Odd_1'], row['Odd_2'])
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0][1 if pred else 0]
            results.append({
                'Player_1': row['Player_1'],
                'Player_2': row['Player_2'],
                'Predicted_Winner': row['Player_1'] if pred else row['Player_2'],
                'Confidence (%)': round(prob*100, 2)
            })
        st.dataframe(pd.DataFrame(results))
