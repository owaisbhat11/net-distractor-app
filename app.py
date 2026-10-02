import streamlit as st
import pandas as pd
import re
from rapidfuzz import fuzz

st.set_page_config(page_title="NTA Distractor Matrix", page_icon="🧠", layout="wide")
st.title("🧠 NTA Distractor Evolution Predictor")
st.markdown("""
Upload a CSV of past papers, and this tool will reverse-engineer the NTA paper-setter's habits 
by finding where an incorrect option (distractor) became a future question.
""")

def clean_text(text):
    if not isinstance(text, str): 
        return ""
    text = re.sub(r"^[\(\[]?[A-Da-d1-4][\)\]\.\-\s]+", "", text)
    text = re.sub(r'[\'\"“”‘’]', '', text)
    return text.strip()

def find_col(columns, patterns):
    for col in columns:
        for pat in patterns:
            if re.search(pat, col, re.IGNORECASE):
                return col
    return None

uploaded_file = st.file_uploader("Upload your PYQ Database (CSV)", type=['csv'])
threshold = st.slider("Match Confidence Threshold (%)", min_value=70, max_value=100, value=82)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Clean column names (strip spaces, lowercase)
        df.columns = [c.strip() for c in df.columns]
        cols = list(df.columns)

        # Smart column detector
        id_col = find_col(cols, [r'^id', r'question.*id', r'q.*id', r'sr.*no', r'sl.*no'])
        session_col = find_col(cols, [r'session', r'exam', r'year', r'cycle', r'date'])
        stem_col = find_col(cols, [r'stem', r'question', r'q_text', r'text'])
        opt_a_col = find_col(cols, [r'option.*[1a]$', r'opt.*[1a]$', r'^a$', r'^1$'])
        opt_b_col = find_col(cols, [r'option.*[2b]$', r'opt.*[2b]$', r'^b$', r'^2$'])
        opt_c_col = find_col(cols, [r'option.*[3c]$', r'opt.*[3c]$', r'^c$', r'^3$'])
        opt_d_col = find_col(cols, [r'option.*[4d]$', r'opt.*[4d]$', r'^d$', r'^4$'])
        ans_col = find_col(cols, [r'correct', r'ans', r'key', r'right'])

        # Fallback assignments if exact patterns are missing
        if not id_col:
            df['question_id'] = [f"Q_{i+1}" for i in range(len(df))]
            id_col = 'question_id'
        if not session_col:
            st.error("Could not detect an Exam Session/Year column. Please ensure your CSV has a 'session' or 'year' column.")
            st.stop()
        if not stem_col or not ans_col:
            st.error("Could not detect Question Text or Correct Answer column. Please check your CSV format.")
            st.stop()

        df[session_col] = df[session_col].astype(str)
        df = df.sort_values(by=[session_col, id_col]).reset_index(drop=True)

        records = []
        for idx, row in df.iterrows():
            correct_val = str(row[ans_col]).strip()
            
            # Resolve correct text based on whether key is a letter/number or direct answer text
            opt_map = {
                'a': clean_text(str(row[opt_a_col])) if opt_a_col else '',
                'b': clean_text(str(row[opt_b_col])) if opt_b_col else '',
                'c': clean_text(str(row[opt_c_col])) if opt_c_col else '',
                'd': clean_text(str(row[opt_d_col])) if opt_d_col else '',
                '1': clean_text(str(row[opt_a_col])) if opt_a_col else '',
                '2': clean_text(str(row[opt_b_col])) if opt_b_col else '',
                '3': clean_text(str(row[opt_c_col])) if opt_c_col else '',
                '4': clean_text(str(row[opt_d_col])) if opt_d_col else ''
            }
            
            key_clean = correct_val.lower()
            if key_clean in opt_map and opt_map[key_clean]:
                correct_text = opt_map[key_clean]
                distractors = [v for k, v in opt_map.items() if k in ['a','b','c','d'] and k != key_clean and v]
            else:
                correct_text = clean_text(correct_val)
                raw_opts = [opt_map['a'], opt_map['b'], opt_map['c'], opt_map['d']]
                distractors = [o for o in raw_opts if o and o.lower() != correct_text.lower()]

            records.append({
                'id': row[id_col],
                'session': row[session_col],
                'stem': clean_text(str(row[stem_col])),
                'correct_text': correct_text,
                'distractors': distractors
            })

        matches = []
        total_q = len(records)
        progress_bar = st.progress(0)

        for i in range(total_q):
            t1_item = records[i]
            for distractor in t1_item['distractors']:
                if len(distractor) < 4 or distractor.lower() in ["none of these", "all of the above"]:
                    continue
                for j in range(i + 1, total_q):
                    t2_item = records[j]
                    if t2_item['session'] <= t1_item['session']:
                        continue
                    
                    score_ans = fuzz.token_sort_ratio(distractor.lower(), t2_item['correct_text'].lower())
                    if score_ans >= threshold:
                        matches.append({
                            "Distractor Term": distractor,
                            "Original Session": t1_item['session'],
                            "Original Question ID": t1_item['id'],
                            "Target Session": t2_item['session'],
                            "Target Question ID": t2_item['id'],
                            "Evolution Type": "Became Answer",
                            "Confidence": f"{score_ans:.1f}%"
                        })
            progress_bar.progress(int((i / total_q) * 100))

        progress_bar.empty()

        if matches:
            st.success(f"Found {len(matches)} recycled distractors!")
            results_df = pd.DataFrame(matches).sort_values(by="Confidence", ascending=False)
            st.dataframe(results_df, use_container_width=True)
            
            csv_export = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Evolution Report (CSV)", data=csv_export, file_name="distractor_evolution_report.csv")
        else:
            st.warning("No matches found. Try lowering the threshold or checking your session dates.")

    except Exception as e:
        st.error(f"Processing error: {str(e)}")
