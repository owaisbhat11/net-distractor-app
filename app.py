import streamlit as st
import pandas as pd
import re
from rapidfuzz import fuzz

# --- UI CONFIGURATION ---
st.set_page_config(page_title="NTA Distractor Matrix", page_icon="🧠", layout="wide")
st.title("🧠 NTA Distractor Evolution Predictor")
st.markdown("""
Upload a CSV of past papers, and this tool will reverse-engineer the NTA paper-setter's habits 
by finding where an incorrect option (distractor) became a future question.
""")

# --- HELPER FUNCTIONS ---
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"^[\(\[]?[A-Da-d1-4][\)\]\.\-\s]+", "", text)
    text = re.sub(r'[\'\"“”‘’]', '', text)
    return text.strip()

def extract_distractors(row):
    correct_key = str(row['correct_key']).strip().lower()
    correct_text = clean_text(row.get(f'option_{correct_key}', ''))
    distractors = [clean_text(row.get(f'option_{k}', '')) for k in ['a', 'b', 'c', 'd'] if k != correct_key]
    return correct_text, [d for d in distractors if d]

# --- MAIN APP LOGIC ---
uploaded_file = st.file_uploader("Upload your PYQ Database (CSV)", type=['csv'])

# Bonus: If they don't upload a file, you could link directly to a public Google Drive CSV file here!
# example_drive_url = "https://docs.google.com/spreadsheets/d/YOUR_FILE_ID/export?format=csv"

threshold = st.slider("Match Confidence Threshold (%)", min_value=70, max_value=100, value=82)

if uploaded_file is not None:
    with st.spinner("Analyzing temporal distractor patterns..."):
        df = pd.read_csv(uploaded_file)
        df['exam_session'] = df['exam_session'].astype(str)
        df = df.sort_values(by=['exam_session', 'question_id']).reset_index(drop=True)
        
        records = []
        for idx, row in df.iterrows():
            correct_text, distractors = extract_distractors(row)
            records.append({
                'id': row['question_id'], 'session': row['exam_session'],
                'stem': clean_text(str(row['stem'])), 'correct_text': correct_text,
                'distractors': distractors
            })

        matches = []
        total_q = len(records)
        
        # Progress bar for the UI
        progress_bar = st.progress(0)
        
        for i in range(total_q):
            t1_item = records[i]
            for distractor in t1_item['distractors']:
                if len(distractor) < 4 or distractor.lower() in ["none of these", "all of the above"]:
                    continue
                for j in range(i + 1, total_q):
                    t2_item = records[j]
                    if t2_item['session'] <= t1_item['session']: continue
                    
                    score_ans = fuzz.token_sort_ratio(distractor.lower(), t2_item['correct_text'].lower())
                    if score_ans >= threshold:
                        matches.append({
                            "Distractor Term": distractor,
                            "Original Appearance": t1_item['session'],
                            "Evolved Appearance": t2_item['session'],
                            "Evolution Type": "Became Answer",
                            "Confidence": f"{score_ans:.1f}%"
                        })
            
            # Update progress bar
            progress_bar.progress(int((i / total_q) * 100))
            
        progress_bar.empty()
        
        if matches:
            st.success(f"Found {len(matches)} recycled distractors!")
            results_df = pd.DataFrame(matches).sort_values(by="Confidence", ascending=False)
            
            # Display interactive table
            st.dataframe(results_df, use_container_width=True)
            
            # Download button
            csv_export = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Full Report (CSV)", data=csv_export, file_name="distractor_matrix.csv")
        else:
            st.warning("No matches found. Try lowering the threshold or checking your data dates.")