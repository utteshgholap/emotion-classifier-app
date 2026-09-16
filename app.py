import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="EmotiSense AI | Emotion Intelligence Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS injection for sleek modern UI)
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .sentiment-badge {
        font-size: 26px;
        font-weight: 700;
        padding: 10px 24px;
        border-radius: 50px;
        display: inline-block;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Ground-truth Emotion Mapping for aug_emotion_clean_data_v2.csv
EMOTIONS = {
    0: {"name": "Anger / Fear", "emoji": "😡", "color": "#ef4444"},
    1: {"name": "Joy", "emoji": "😄", "color": "#10b981"},
    2: {"name": "Sadness", "emoji": "😢", "color": "#3b82f6"},
    3: {"name": "Love", "emoji": "❤️", "color": "#ec4899"},
    4: {"name": "Surprise", "emoji": "😲", "color": "#f59e0b"}
}

# Cache model artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('emotion_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}. Ensure emotion_model.pkl and tfidf_vectorizer.pkl are in the directory.")
    st.stop()

# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar Controls & Information
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8649/8649595.png", width=70)
    st.title("EmotiSense AI")
    st.caption("NLP Architecture: TF-IDF Vectorizer + Multinomial Logistic Regression")
    
    st.markdown("---")
    st.subheader("💡 Quick Sample Texts")
    sample_text = st.selectbox(
        "Load an example prompt:",
        (
            "-- Select an example --",
            "I am absolutely ecstatic about the promotion I received today!",
            "I will kill you and break everything you care about.",
            "I feel completely broken, empty, and isolated inside.",
            "You are the most wonderful person I have ever known, my love.",
            "I was stunned to find out about the secret ceremony yesterday!"
        )
    )
    
    st.markdown("---")
    if st.button("🗑️ Clear Prediction History", use_container_width=True):
        st.session_state.history = []
        st.success("History cleared!")

# Main Tabs Navigation
tab1, tab2, tab3 = st.tabs(["🎯 Real-Time Analysis", "📁 Batch File Processing", "📜 Prediction Logs"])

# TAB 1: Single Sentence Real-Time Classifier
with tab1:
    col_input, col_output = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.subheader("Text Input")
        default_val = "" if sample_text == "-- Select an example --" else sample_text
        user_input = st.text_area(
            "Enter statement to evaluate:",
            value=default_val,
            height=180,
            placeholder="Type your message, review, or statement here..."
        )
        
        analyze_btn = st.button("🚀 Analyze Emotion", type="primary", use_container_width=True)

    with col_output:
        st.subheader("Inference Result")
        if analyze_btn and user_input.strip():
            # Vectorize & Infer
            vec_input = vectorizer.transform([user_input])
            pred_id = model.predict(vec_input)[0]
            probs = model.predict_proba(vec_input)[0]
            
            emo_info = EMOTIONS[pred_id]
            max_conf = probs[pred_id] * 100
            
            # Record in history
            st.session_state.history.append({
                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                "Text": user_input[:60] + ("..." if len(user_input) > 60 else ""),
                "Predicted Emotion": f"{emo_info['name']} {emo_info['emoji']}",
                "Confidence": f"{max_conf:.2f}%"
            })

            # Prediction Card
            st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 14px; color: #64748b; text-transform: uppercase; letter-spacing: 1px;">Primary Emotion Detected</div>
                    <div class="sentiment-badge" style="background-color: {emo_info['color']}20; color: {emo_info['color']};">
                        {emo_info['emoji']} {emo_info['name']}
                    </div>
                    <div style="font-size: 16px; color: #334155; font-weight: 500;">
                        Confidence: <b>{max_conf:.2f}%</b>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Interactive Confidence Breakdown Chart
            chart_data = pd.DataFrame({
                "Emotion": [f"{v['emoji']} {v['name']}" for k, v in EMOTIONS.items()],
                "Probability": probs,
                "Color": [v['color'] for k, v in EMOTIONS.items()]
            }).sort_values(by="Probability", ascending=True)

            fig = px.bar(
                chart_data,
                x="Probability",
                y="Emotion",
                orientation='h',
                text=chart_data["Probability"].apply(lambda x: f"{x*100:.1f}%"),
                color="Emotion",
                color_discrete_sequence=chart_data["Color"].tolist()
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=25, b=10),
                height=260,
                xaxis=dict(showgrid=False, range=[0, 1]),
                yaxis=dict(title=None)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif analyze_btn and not user_input.strip():
            st.warning("Please enter text or choose an example from the sidebar.")

# TAB 2: Batch CSV Upload
with tab2:
    st.subheader("Batch Dataset Classification")
    st.write("Upload a CSV file containing a column of texts to run bulk inference and generate distributions.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(batch_df.head(4), use_container_width=True)
        
        target_col = st.selectbox("Select text column for prediction:", batch_df.columns)
        
        if st.button("⚡ Process Batch Predictions", type="primary"):
            with st.spinner("Classifying batch rows..."):
                clean_series = batch_df[target_col].fillna("")
                vec_batch = vectorizer.transform(clean_series)
                batch_preds = model.predict(vec_batch)
                
                batch_df['Predicted_Emotion'] = [f"{EMOTIONS[i]['name']} {EMOTIONS[i]['emoji']}" for i in batch_preds]
                
                st.success(f"Processed {len(batch_df)} rows successfully!")
                
                # Distribution Chart
                dist_counts = batch_df['Predicted_Emotion'].value_counts().reset_index()
                dist_counts.columns = ['Emotion', 'Count']
                fig_pie = px.pie(dist_counts, names='Emotion', values='Count', hole=0.4, title="Distribution of Predicted Emotions")
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Download Result
                csv_download = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Enriched CSV", data=csv_download, file_name="emotion_classified_output.csv", mime="text/csv")

# TAB 3: History & Logging
with tab3:
    st.subheader("Real-Time Session History")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No predictions made in this session yet.")