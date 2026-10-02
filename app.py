import streamlit as st
import pandas as pd
import re
from rapidfuzz import fuzz

st.set_page_config(page_title="NTA Distractor Matrix", page_icon="🧠", layout="wide")
st.title("🧠 NTA Distractor Evolution Predictor")
st.markdown("""
Upload a CSV of past papers. This tool filters out standard boilerplates (e.g., 'Statement I and II', 'A, B only')
and isolates **genuine literary terms, authors, and conceptual distractors** that evolved across exam cycles.
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

def is_boilerplate_or_code(text):
    """Filters out structural exam boilerplate, statement pairs, and letter combinations."""
    clean = text.lower().strip()
    
    # 1. Structural letter patterns like "A, B and C only", "(A), (B), (C)", "B, C and E only"
    if re.search(r'\b[a-e]\s*(?:,|and|\&)\s*[a-e]', clean):
        return True
    if re.search(r'\b(?:only|both)\b', clean) and re.search(r'\b[a-e]\b', clean):
        return True
        
    # 2. Stripped down alphanumeric codes (e.g., 'ABCDE', 'I, II, III')
    compact = re.sub(r'[\s,\.\-\(\)\:\;]', '', clean)
    if re.fullmatch(r'[a-e1-5ivx]+', compact):
        return True

    # 3. Statement / Assertion boilerplates
    boilerplate_phrases = [
        "statement i", "statement ii", "both statement",
        "assertion", "reason", "is correct", "is incorrect",
        "is true", "is false", "not correct", "all of the above",
        "none of the above", "none of these"
    ]
    if any(phrase in clean for phrase in boilerplate_phrases):
        return True
        
    # 4. Too short or empty
    if len(clean) < 4:
        return True

    return False

def parse_session_date(session_str):
    """Extract chronological order; handles explicit dates, years, and fallback indexing."""
    s = str(session_str)
    year_match = re.search(r'\b(20\d{2})\b', s)
    year = int(year_match.group(1)) if year_match else 2000
    
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    month = 1
    for m_name, m_val in month_map.items():
        if re.search(m_name, s, re.IGNORECASE):
            month = m_val
            break
            
    return (year * 100) + month

uploaded_file = st.file_uploader("Upload your PYQ Database (CSV)", type=['csv'])
threshold = st.slider("Match Confidence Threshold (%)", min_value=75, max_value=100, value=84)

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
                # Filter out boilerplates, codes, and generic combinations
                if is_boilerplate_or_code(distractor):
                    continue

                for j in range(i + 1, total_q):
                    t2_item = records[j]

                    # Enforce strict forward progression
                    if t2_item['sort_date'] <= t1_item['sort_date']:
                        continue
                    
                    if is_boilerplate_or_code(t2_item['correct_text']):
                        continue

                    # Direct option evolution (Distractor in A -> Answer in B)
                    score_ans = fuzz.token_sort_ratio(distractor.lower(), t2_item['correct_text'].lower())
                    if score_ans >= threshold:
                        matches.append({
                            "Matched Literary Entity": distractor,
                            "Original Exam (Cycle A)": t1_item['session'],
                            "Target Exam (Cycle B)": t2_item['session'],
                            "Confidence": f"{score_ans:.1f}%",
                            "Cycle A Stem": t1_item['stem'],
                            "Cycle B Stem": t2_item['stem'],
                            "Cycle B Answer": t2_item['correct_text']
                        })

            progress_bar.progress(int((i / total_q) * 100))

        progress_bar.empty()

        if matches:
            st.success(f"Isolated {len(matches)} genuine literary distractor evolutions!")
            results_df = pd.DataFrame(matches).sort_values(by="Confidence", ascending=False)
            st.dataframe(results_df, use_container_width=True)
            
            csv_export = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Curated Report (CSV)", data=csv_export, file_name="genuine_distractor_evolutions.csv")
        else:
            st.warning("No matches found with current settings. Try adjusting the threshold slightly.")

    except Exception as e:
        st.error(f"Processing error: {str(e)}")
