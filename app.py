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

def is_trivial_code(text):
    """Filter out options that are just answer codes or matching sequences like 'D, B, C, A, E'."""
    clean = re.sub(r'[\s,\.\-\(\)\:]', '', text)
    # Checks if text consists only of option labels, numbers, or roman numerals
    if re.fullmatch(r'[A-Ea-e1-5ivxIVX]+', clean):
        return True
    return False

def parse_session_date(session_str):
    """Extract a sortable year and month from raw session strings like 'Dec 2025' or 'Mar 2023'."""
    s = str(session_str)
    # Look for a 4-digit year (e.g. 2020, 2023, 2025, 2026)
    year_match = re.search(r'\b(20\d{2})\b', s)
    year = int(year_match.group(1)) if year_match else 2000
    
    # Map month name to integer
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    month = 1
    for m_name, m_val in month_map.items():
        if re.search(m_name, s, re.IGNORECASE):
            month = m_val
            break
            
    return (year * 100) + month  # e.g., 202303 for March 2023, 202512 for Dec 2025

uploaded_file = st.file_uploader("Upload your PYQ Database (CSV)", type=['csv'])
threshold = st.slider("Match Confidence Threshold (%)", min_value=70, max_value=100, value=82)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip() for c in df.columns]
        cols = list(df.columns)

        id_col = find_col(cols, [r'^id', r'question.*id', r'q.*id', r'sr.*no', r'sl.*no'])
        session_col = find_col(cols, [r'session', r'exam', r'year', r'cycle', r'date'])
        stem_col = find_col(cols, [r'stem', r'question', r'q_text', r'text'])
        opt_a_col = find_col(cols, [r'option.*[1a]$', r'opt.*[1a]$', r'^a$', r'^1$'])
        opt_b_col = find_col(cols, [r'option.*[2b]$', r'opt.*[2b]$', r'^b$', r'^2$'])
        opt_c_col = find_col(cols, [r'option.*[3c]$', r'opt.*[3c]$', r'^c$', r'^3$'])
        opt_d_col = find_col(cols, [r'option.*[4d]$', r'opt.*[4d]$', r'^d$', r'^4$'])
        ans_col = find_col(cols, [r'correct', r'ans', r'key', r'right'])

        if not id_col:
            df['question_id'] = [f"Q_{i+1}" for i in range(len(df))]
            id_col = 'question_id'
        if not session_col or not stem_col or not ans_col:
            st.error("Missing essential columns. Ensure session, question, and answer columns are present.")
            st.stop()

        # Generate a numerical sort key for true chronological order
        df['sort_date'] = df[session_col].apply(parse_session_date)
        df = df.sort_values(by=['sort_date', id_col]).reset_index(drop=True)

        records = []
        for idx, row in df.iterrows():
            correct_val = str(row[ans_col]).strip()
            
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
                'sort_date': row['sort_date'],
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
                # Skip junk, codes like 'A, B, C', or single short words
                if len(distractor) < 4 or is_trivial_code(distractor) or distractor.lower() in ["none of these", "all of the above"]:
                    continue

                for j in range(i + 1, total_q):
                    t2_item = records[j]

                    # Strictly enforce forward-in-time matching
                    if t2_item['sort_date'] <= t1_item['sort_date']:
                        continue
                    
                    score_ans = fuzz.token_sort_ratio(distractor.lower(), t2_item['correct_text'].lower())
                    if score_ans >= threshold:
                        matches.append({
                            "Distractor Term": distractor,
                            "Original Session (Cycle A)": t1_item['session'],
                            "Target Session (Cycle B)": t2_item['session'],
                            "Confidence": f"{score_ans:.1f}%",
                            "Original Question": t1_item['stem'],
                            "Target Answer": t2_item['correct_text']
                        })

            progress_bar.progress(int((i / total_q) * 100))

        progress_bar.empty()

        if matches:
            st.success(f"Found {len(matches)} genuine recycled distractors!")
            results_df = pd.DataFrame(matches).sort_values(by="Confidence", ascending=False)
            st.dataframe(results_df, use_container_width=True)
            
            csv_export = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Report (CSV)", data=csv_export, file_name="cleaned_distractor_report.csv")
        else:
            st.warning("No matches found. Try lowering the threshold slider slightly.")

    except Exception as e:
        st.error(f"Processing error: {str(e)}")
