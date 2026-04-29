# =============================================================================
# paper_pipeline.py  --  Stages P1 . P2 . P3 . P4A . P4B . P5
#
# Pre-operative Text -> Post-operative Diagnosis Prediction
#
# P1  : Merge post-op targets onto cleaned features, encode labels
# P2  : Bio_ClinicalBERT encoding of scheduled_procedure (768-d)
# P3  : Tabular feature matrices + fold-isolated FAISS indexes
# P4A : Task A -- ICD-10 classification  (LightGBM, XGBoost, baseline)
# P4B : Task B -- RAG + Llama-3 operative_dx generation
# P5  : Ablation study (text-only, tabular-only, TF-IDF, RAG-K variants)
# =============================================================================

# =============================================================================
# CONFIG
# =============================================================================

RAW_CSV       = './data/casetime.csv'
SOURCE_DB     = './data/surgical_data.db'   # existing Clean table from pipeline.py
PAPER_DB      = './data/paper_data.db'
BERT_CACHE    = './data/paper_bert_cache.npy'
FAISS_DIR     = './data/paper_faiss'
RESULTS_DIR   = './results/paper'

CLEAN_TABLE   = 'Clean'
PAPER_TABLE   = 'PaperClean'
FOLD_TABLE    = 'fold_indices'

N_SPLITS      = 5
RANDOM_STATE  = 42

# ICD classes with fewer than this many examples are grouped as 'OTHER'
ICD_MIN_COUNT = 50

# Columns to use as structured (pre-op only) features
STRUCT_COLS   = [
    'age_at_discharge', 'sex', 'avg_BMI', 'OR_team_size',
    'month_of_year', 'day_of_week', 'scheduled_start_hour',
    'scheduled_duration', 'ASA_score',
    'first_scheduled_case_of_day_status',
    'last_scheduled_case_of_day_status',
    'OR_trip_sequence', 'primary_procedure_status',
]
CAT_COLS      = ['case_service', 'surgical_location', 'anesthetic_type']
TEXT_COL      = 'scheduled_procedure'
TARGET_A      = 'icd_label'        # encoded most_responsible_dx
TARGET_B      = 'operative_dx_norm' # normalised operative_dx

# ClinicalBERT
CLINBERT_ID   = 'emilyalsentzer/Bio_ClinicalBERT'
BERT_BATCH    = 64
BERT_MAXLEN   = 64

# LightGBM
LGB_PARAMS = {
    'objective':        'multiclass',
    'metric':           'multi_logloss',
    'learning_rate':    0.1,
    'num_leaves':       31,
    'min_data_in_leaf': 30,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq':     5,
    'verbose':          -1,
    'n_jobs':           -1,
    'num_threads':      8,
}
LGB_ROUNDS    = 200
LGB_EARLY     = 20

# Logistic Regression (replaces XGBoost for multi-class speed)
LR_MAX_ITER   = 1000
LR_C          = 1.0

# RAG
RAG_K             = 5       # neighbours to retrieve
RAG_SAMPLE_PER_FOLD = 100   # test cases to evaluate per fold (Ollama is slow)
OLLAMA_MODEL      = 'llama3:latest'
OLLAMA_TIMEOUT    = 30      # seconds per call

# =============================================================================
# IMPORTS
# =============================================================================

import os, sys, re, sqlite3, time, warnings, json, pickle
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FAISS_DIR,   exist_ok=True)

# =============================================================================
# UTILITIES
# =============================================================================

def sep(title='', w=72):
    line = '=' * w
    print(f'\n{line}')
    if title:
        print(f'  {title}')
        print(line)

def topk_accuracy(y_true, proba, k):
    """Fraction of samples where true label is in top-k predicted."""
    top = np.argsort(proba, axis=1)[:, -k:]
    return float(np.mean([y_true[i] in top[i] for i in range(len(y_true))]))

def save_results(rows, path):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f'  Saved -> {path}')

# =============================================================================
# STAGE P1 -- DATA PREPARATION
# =============================================================================

def _p1_done():
    if not os.path.exists(PAPER_DB): return False
    try:
        with sqlite3.connect(PAPER_DB) as c:
            n = c.execute(f"SELECT COUNT(*) FROM {PAPER_TABLE}").fetchone()[0]
            return n > 0
    except: return False

def run_p1():
    if _p1_done():
        print('  [>>] Stage P1 already done. Skipping.')
        return
    sep('STAGE P1 -- DATA PREPARATION')

    # ── load existing Clean features (already processed by pipeline.py) ───
    with sqlite3.connect(SOURCE_DB) as conn:
        clean = pd.read_sql(f'SELECT * FROM {CLEAN_TABLE}', conn)
    print(f'  Loaded Clean table: {clean.shape}')

    # ── load post-op text targets from raw CSV ────────────────────────────
    csv = pd.read_csv(RAW_CSV)
    csv.columns = csv.columns.str.strip()
    csv['case_id'] = csv['case_id'].astype(int)
    targets = csv[['case_id', 'most_responsible_dx', 'operative_dx']].copy()
    targets['most_responsible_dx'] = (
        targets['most_responsible_dx'].astype(str).str.strip()
    )
    targets['operative_dx_norm'] = (
        targets['operative_dx'].astype(str).str.lower().str.strip()
    )

    # ── merge ─────────────────────────────────────────────────────────────
    df = clean.merge(targets[['case_id','most_responsible_dx','operative_dx_norm']],
                     on='case_id', how='inner')
    print(f'  After merge: {df.shape}')

    # ── drop post-op feature columns (leakage prevention) ─────────────────
    POSTOP = ['actual_casetime_minutes','procedure_minutes','procedure_time',
              'induction_time','emergence_time','or_entry_hour',
              'surgery_encounter_inpatient']
    df.drop(columns=[c for c in POSTOP if c in df.columns], inplace=True)
    print(f'  Dropped post-op columns: {POSTOP}')

    # ── filter uninformative operative_dx ─────────────────────────────────
    BAD_OPDX = {'unknown operativedx','unknown','n/a','na','nan','none',''}
    before = len(df)
    df = df[~df['operative_dx_norm'].isin(BAD_OPDX)].copy()
    print(f'  Filtered uninformative operative_dx: -{before - len(df):,} rows')

    # ── ICD label encoding (rare classes -> "OTHER") ──────────────────────
    icd_counts = df['most_responsible_dx'].value_counts()
    rare        = set(icd_counts[icd_counts < ICD_MIN_COUNT].index)
    df['most_responsible_dx_clean'] = df['most_responsible_dx'].apply(
        lambda x: 'OTHER' if x in rare else x
    )
    n_classes = df['most_responsible_dx_clean'].nunique()
    print(f'  ICD classes (>= {ICD_MIN_COUNT} occurrences): {n_classes}  '
          f'(collapsed {len(rare):,} rare codes into OTHER)')

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df[TARGET_A] = le.fit_transform(df['most_responsible_dx_clean'])
    # save label encoder
    with open(f'{RESULTS_DIR}/icd_label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print(f'  Label encoder saved. Classes: {len(le.classes_)}')

    # ── scheduled_duration missing check ──────────────────────────────────
    if 'scheduled_duration' not in df.columns:
        print('  WARNING: scheduled_duration not found — deriving from CSV')
        csv2 = pd.read_csv(RAW_CSV)
        csv2.columns = csv2.columns.str.strip()
        csv2['scheduled_duration'] = (
            pd.to_datetime(csv2['scheduled_end_dttm']) -
            pd.to_datetime(csv2['scheduled_start_dttm'])
        ).dt.total_seconds() / 60
        df = df.merge(
            csv2[['case_id','scheduled_duration']].rename(
                columns={'case_id':'case_id'}),
            on='case_id', how='left')

    # ── final shape & save ────────────────────────────────────────────────
    df.reset_index(drop=True, inplace=True)
    print(f'\n  Final shape: {df.shape}')
    print(f'  Pre-op feature cols: {STRUCT_COLS + CAT_COLS + [TEXT_COL]}')
    print(f'  Target A (ICD):      {TARGET_A}  ({n_classes} classes)')
    print(f'  Target B (free-text):{TARGET_B}  '
          f'({df[TARGET_B].nunique():,} unique phrases)')

    with sqlite3.connect(PAPER_DB) as conn:
        df.to_sql(PAPER_TABLE, conn, if_exists='replace', index=False)
    print(f'  Saved {PAPER_TABLE} -> {PAPER_DB}')
    print('  [OK] Stage P1 complete.')

# =============================================================================
# STAGE P2 -- CLINICALBERT ENCODING
# =============================================================================

def _p2_done():
    return os.path.exists(BERT_CACHE)

def run_p2():
    if _p2_done():
        print('  [>>] Stage P2 already done. Skipping.')
        return
    sep('STAGE P2 -- CLINICALBERT ENCODING')

    import torch
    from transformers import AutoTokenizer, AutoModel

    with sqlite3.connect(PAPER_DB) as conn:
        texts = pd.read_sql(
            f'SELECT case_id, {TEXT_COL} FROM {PAPER_TABLE}', conn
        )
    print(f'  Encoding {len(texts):,} texts with {CLINBERT_ID}')

    tokenizer = AutoTokenizer.from_pretrained(CLINBERT_ID)
    model     = AutoModel.from_pretrained(CLINBERT_ID)
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval().to(device)
    print(f'  Device: {device}')

    text_list = texts[TEXT_COL].fillna('').tolist()
    all_embs  = []
    n_batches = (len(text_list) + BERT_BATCH - 1) // BERT_BATCH

    t0 = time.time()
    for i in range(0, len(text_list), BERT_BATCH):
        batch = text_list[i:i + BERT_BATCH]
        with torch.no_grad():
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=BERT_MAXLEN, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            emb = out.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token 768-d
        all_embs.append(emb)
        if (i // BERT_BATCH) % 100 == 0:
            elapsed = time.time() - t0
            pct     = (i + BERT_BATCH) / len(text_list)
            eta     = (elapsed / max(pct, 1e-6)) * (1 - pct)
            print(f'  Batch {i//BERT_BATCH+1}/{n_batches}  '
                  f'({i+BERT_BATCH:,}/{len(text_list):,})  '
                  f'ETA {eta/60:.1f} min', flush=True)

    embs = np.vstack(all_embs).astype(np.float32)
    np.save(BERT_CACHE, embs)
    print(f'  Saved embeddings {embs.shape} -> {BERT_CACHE}')
    print(f'  Total time: {(time.time()-t0)/60:.1f} min')
    print('  [OK] Stage P2 complete.')

# =============================================================================
# STAGE P3 -- FEATURE MATRICES + FAISS INDEXES
# =============================================================================

def _p3_done():
    return all(
        os.path.exists(f'{FAISS_DIR}/fold{f}_index.faiss')
        and os.path.exists(f'{FAISS_DIR}/fold{f}_meta.pkl')
        for f in range(N_SPLITS)
    )

def _build_tabular(df_tr, df_va, cat_cols, struct_cols):
    """Fold-wise: fit imputer + one-hot on train, transform both."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    import scipy.sparse as sp

    # numeric
    num_tr = df_tr[struct_cols].values.astype(np.float32)
    num_va = df_va[struct_cols].values.astype(np.float32)
    imp    = SimpleImputer(strategy='median')
    num_tr = imp.fit_transform(num_tr)
    num_va = imp.transform(num_va)

    # categorical one-hot
    cat_tr = df_tr[cat_cols].fillna('MISSING').astype(str)
    cat_va = df_va[cat_cols].fillna('MISSING').astype(str)
    ohe    = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe_tr = ohe.fit_transform(cat_tr)
    ohe_va = ohe.transform(cat_va)

    X_tr = np.hstack([num_tr, ohe_tr]).astype(np.float32)
    X_va = np.hstack([num_va, ohe_va]).astype(np.float32)
    return X_tr, X_va, imp, ohe

def run_p3():
    if _p3_done():
        print('  [>>] Stage P3 already done. Skipping.')
        return
    sep('STAGE P3 -- FEATURE MATRICES + FAISS INDEXES')
    import faiss

    with sqlite3.connect(PAPER_DB) as conn:
        df = pd.read_sql(f'SELECT * FROM {PAPER_TABLE}', conn)
    bert_embs = np.load(BERT_CACHE).astype(np.float32)
    print(f'  Loaded {len(df):,} rows, BERT shape {bert_embs.shape}')

    # fold indices from existing pipeline folds
    with sqlite3.connect(SOURCE_DB) as conn:
        folds_src = pd.read_sql(f'SELECT * FROM {FOLD_TABLE}', conn)

    # re-generate folds on paper data by case_id match
    from sklearn.model_selection import KFold
    kf  = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    idx = np.arange(len(df))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(idx)):
        t0 = time.time()
        print(f'\n  Fold {fold} — train {len(tr_idx):,}  val {len(va_idx):,}')

        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_va = df.iloc[va_idx].reset_index(drop=True)

        # tabular features (fold-wise, no leakage)
        X_tab_tr, X_tab_va, imp, ohe = _build_tabular(
            df_tr, df_va, CAT_COLS, STRUCT_COLS)

        # BERT embeddings
        bert_tr = bert_embs[tr_idx]
        bert_va = bert_embs[va_idx]

        # full feature matrix = BERT + tabular
        X_tr = np.hstack([bert_tr, X_tab_tr])
        X_va = np.hstack([bert_va, X_tab_va])
        print(f'  Feature dim: {X_tr.shape[1]}  '
              f'(768 BERT + {X_tab_tr.shape[1]} tabular)')

        # FAISS index (train fold only — no leakage)
        # L2-normalise for cosine similarity via inner product
        norms   = np.linalg.norm(bert_tr, axis=1, keepdims=True).clip(1e-8)
        bert_tr_norm = (bert_tr / norms).astype(np.float32)
        index   = faiss.IndexFlatIP(bert_tr_norm.shape[1])
        index.add(bert_tr_norm)
        faiss.write_index(index, f'{FAISS_DIR}/fold{fold}_index.faiss')

        # meta: save everything needed for P4A and P4B
        meta = {
            'tr_idx':    tr_idx,
            'va_idx':    va_idx,
            'X_tr':      X_tr,
            'X_va':      X_va,
            'bert_tr':   bert_tr,
            'bert_va':   bert_va,
            'y_A_tr':    df_tr[TARGET_A].values,
            'y_A_va':    df_va[TARGET_A].values,
            'opdx_tr':   df_tr[TARGET_B].values,
            'opdx_va':   df_va[TARGET_B].values,
            'raw_tr':    df_tr[[TEXT_COL]+STRUCT_COLS+CAT_COLS+
                                [TARGET_A,TARGET_B]].reset_index(drop=True),
            'raw_va':    df_va[[TEXT_COL]+STRUCT_COLS+CAT_COLS+
                                [TARGET_A,TARGET_B]].reset_index(drop=True),
            'imp':       imp,
            'ohe':       ohe,
        }
        with open(f'{FAISS_DIR}/fold{fold}_meta.pkl', 'wb') as fh:
            pickle.dump(meta, fh)
        print(f'  Fold {fold} done in {time.time()-t0:.1f}s')

    print('\n  [OK] Stage P3 complete.')

# =============================================================================
# STAGE P4A -- TASK A: ICD CLASSIFICATION
# =============================================================================

def _p4a_done():
    return os.path.exists(f'{RESULTS_DIR}/p4a_results.csv')

def run_p4a():
    if _p4a_done():
        print('  [>>] Stage P4A already done. Skipping.')
        return
    sep('STAGE P4A -- TASK A: ICD-10 CLASSIFICATION')
    import lightgbm as lgb
    from sklearn.preprocessing import normalize as sk_normalize

    results = []

    for fold in range(N_SPLITS):
        t0 = time.time()
        print(f'\n--- Fold {fold} ---')
        with open(f'{FAISS_DIR}/fold{fold}_meta.pkl', 'rb') as fh:
            m = pickle.load(fh)

        X_tr, X_va   = m['X_tr'], m['X_va']
        y_tr, y_va   = m['y_A_tr'], m['y_A_va']
        n_cls        = int(max(y_tr.max(), y_va.max())) + 1
        raw_tr       = m['raw_tr']
        raw_va       = m['raw_va']

        # ── Baseline: mode per scheduled_procedure (train-fold only) ──────
        modal_map = (raw_tr.groupby(TEXT_COL)[TARGET_A]
                     .agg(lambda x: x.mode()[0]))
        proc_va   = raw_va[TEXT_COL].values
        b_preds   = np.array([modal_map.get(p, y_tr[0]) for p in proc_va])
        b_top1    = float(np.mean(b_preds == y_va))
        results.append({'fold':fold,'model':'Baseline(mode)',
                        'top1':b_top1,'top3':b_top1,'top5':b_top1,'time_s':0})
        print(f'  Baseline (fold-aware) top-1: {b_top1:.4f}')

        # ── SGD Classifier on L2-normalised BERT features (fast baseline) ─
        from sklearn.linear_model import SGDClassifier
        from sklearn.calibration import CalibratedClassifierCV
        t1     = time.time()
        X_tr_n = sk_normalize(X_tr[:, :768])
        X_va_n = sk_normalize(X_va[:, :768])
        sgd    = SGDClassifier(loss='modified_huber', max_iter=200,
                               random_state=RANDOM_STATE, n_jobs=-1, tol=1e-3)
        sgd.fit(X_tr_n, y_tr)
        proba_lr = sgd.predict_proba(X_va_n)
        top1_lr  = float(np.mean(proba_lr.argmax(1) == y_va))
        top3_lr  = topk_accuracy(y_va, proba_lr, 3)
        top5_lr  = topk_accuracy(y_va, proba_lr, 5)
        elapsed  = time.time() - t1
        results.append({'fold':fold,'model':'SGD(BERT)',
                        'top1':top1_lr,'top3':top3_lr,'top5':top5_lr,
                        'time_s':round(elapsed,1)})
        print(f'  SGD(BERT)  top1={top1_lr:.4f}  top3={top3_lr:.4f}  '
              f'top5={top5_lr:.4f}  ({elapsed:.0f}s)')

        # ── LightGBM (BERT + Tabular) ─────────────────────────────────────
        params = {**LGB_PARAMS, 'num_class': n_cls}
        lgb_tr = lgb.Dataset(X_tr, label=y_tr)
        lgb_va = lgb.Dataset(X_va, label=y_va, reference=lgb_tr)
        t1     = time.time()
        bst    = lgb.train(params, lgb_tr,
                           num_boost_round=LGB_ROUNDS,
                           valid_sets=[lgb_va],
                           callbacks=[
                               lgb.early_stopping(LGB_EARLY, verbose=False),
                               lgb.log_evaluation(period=100),
                           ])
        proba  = bst.predict(X_va)
        top1   = float(np.mean(proba.argmax(1) == y_va))
        top3   = topk_accuracy(y_va, proba, 3)
        top5   = topk_accuracy(y_va, proba, 5)
        elapsed = time.time() - t1
        bst.save_model(f'{RESULTS_DIR}/lgb_fold{fold}.txt')
        results.append({'fold':fold,'model':'LightGBM(BERT+Tab)',
                        'top1':top1,'top3':top3,'top5':top5,
                        'time_s':round(elapsed,1)})
        print(f'  LightGBM(BERT+Tab) top1={top1:.4f}  top3={top3:.4f}  '
              f'top5={top5:.4f}  ({elapsed:.0f}s)')
        print(f'  Fold {fold} total: {time.time()-t0:.0f}s')

    # ── Summary ───────────────────────────────────────────────────────────
    res_df = pd.DataFrame(results)
    save_results(results, f'{RESULTS_DIR}/p4a_results.csv')
    sep('TASK A SUMMARY')
    summary = res_df.groupby('model')[['top1','top3','top5']].mean()
    print(summary.round(4).to_string())
    print('\n  [OK] Stage P4A complete.')

# =============================================================================
# STAGE P4B -- TASK B: RAG + LLAMA-3 GENERATION
# =============================================================================

def _p4b_done():
    return os.path.exists(f'{RESULTS_DIR}/p4b_results.csv')

def _rag_prompt(row, neighbours):
    """Build Llama-3 prompt from pre-op row + retrieved neighbours."""
    sex_str = 'Male' if row.get('sex', 0) == 1 else 'Female'
    age     = row.get('age_at_discharge', 'N/A')
    bmi     = row.get('avg_BMI', 'N/A')
    asa     = row.get('ASA_score', 'N/A')
    svc     = row.get('case_service', 'N/A')
    anes    = row.get('anesthetic_type', 'N/A')
    dur     = row.get('scheduled_duration', 'N/A')
    proc    = row.get(TEXT_COL, '')

    nb_lines = '\n'.join(
        f"  {i+1}. {nb['proc']} -> {nb['opdx']}"
        for i, nb in enumerate(neighbours)
    )

    prompt = (
        "You are a surgical clinical decision support system.\n"
        "Given pre-operative patient data and similar past cases, "
        "predict the operative diagnosis.\n"
        "Reply with ONLY the diagnosis phrase (2-8 words). No explanation.\n\n"
        f"Patient:\n"
        f"  Scheduled procedure : {proc}\n"
        f"  Age {age}  Sex {sex_str}  BMI {bmi}  ASA {asa}\n"
        f"  Service: {svc}  Anesthetic: {anes}  "
        f"Scheduled duration: {dur} min\n\n"
        f"Similar past cases (procedure -> operative diagnosis):\n"
        f"{nb_lines}\n\n"
        "Operative diagnosis:"
    )
    return prompt

def run_p4b():
    if _p4b_done():
        print('  [>>] Stage P4B already done. Skipping.')
        return
    sep('STAGE P4B -- TASK B: RAG + LLAMA-3 GENERATION')

    import faiss as faiss_lib
    import ollama
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer as rouge_module
    from sentence_transformers import SentenceTransformer

    scorer_rouge = rouge_module.RougeScorer(['rougeL'], use_stemmer=False)
    st_model     = SentenceTransformer('all-MiniLM-L6-v2')
    smoother     = SmoothingFunction().method1

    # test Ollama connection
    try:
        resp = ollama.generate(model=OLLAMA_MODEL, prompt='hi', options={'num_predict':3})
        print(f'  Ollama {OLLAMA_MODEL} OK')
    except Exception as e:
        print(f'  ERROR: Ollama not reachable: {e}')
        print('  Make sure Ollama is running: ollama serve')
        return

    all_rows = []

    for fold in range(N_SPLITS):
        t0 = time.time()
        print(f'\n--- Fold {fold} ---')

        with open(f'{FAISS_DIR}/fold{fold}_meta.pkl', 'rb') as fh:
            m = pickle.load(fh)

        index = faiss_lib.read_index(f'{FAISS_DIR}/fold{fold}_index.faiss')

        raw_tr = m['raw_tr']   # training rows (used as retrieval corpus)
        raw_va = m['raw_va']   # validation rows (test cases)
        bert_va = m['bert_va'].astype(np.float32)

        # sample test cases
        rng      = np.random.default_rng(RANDOM_STATE + fold)
        n_sample = min(RAG_SAMPLE_PER_FOLD, len(raw_va))
        sample_idx = rng.choice(len(raw_va), n_sample, replace=False)

        bleus, rouges, cosines = [], [], []

        for j, vi in enumerate(sample_idx):
            row     = raw_va.iloc[vi].to_dict()
            ref     = str(row[TARGET_B])
            if not ref or ref in {'nan','none',''}:
                continue

            # FAISS retrieve
            q = bert_va[vi:vi+1]
            q_norm = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-8)
            _, nn_idx = index.search(q_norm, RAG_K)
            neighbours = [
                {'proc': raw_tr.iloc[ni][TEXT_COL],
                 'opdx': raw_tr.iloc[ni][TARGET_B]}
                for ni in nn_idx[0] if ni < len(raw_tr)
            ]

            # Llama-3 generation
            prompt = _rag_prompt(row, neighbours)
            try:
                resp = ollama.generate(
                    model=OLLAMA_MODEL,
                    prompt=prompt,
                    options={'num_predict': 30, 'temperature': 0.1}
                )
                pred = resp['response'].strip().split('\n')[0].strip().lower()
            except Exception as e:
                pred = ''
                print(f'  Ollama error at sample {j}: {e}')

            # metrics
            ref_tok  = ref.split()
            pred_tok = pred.split()
            bleu  = sentence_bleu([ref_tok], pred_tok, smoothing_function=smoother) \
                    if pred_tok else 0.0
            rouge = scorer_rouge.score(ref, pred)['rougeL'].fmeasure \
                    if pred else 0.0

            # cosine similarity via SentenceTransformer
            if pred:
                embs_pair = st_model.encode([ref, pred],
                                            normalize_embeddings=True)
                cosine = float(np.dot(embs_pair[0], embs_pair[1]))
            else:
                cosine = 0.0

            bleus.append(bleu); rouges.append(rouge); cosines.append(cosine)
            all_rows.append({
                'fold': fold, 'sample_idx': int(vi),
                'scheduled_procedure': row.get(TEXT_COL,''),
                'reference': ref, 'prediction': pred,
                'bleu': round(bleu,4), 'rouge_l': round(rouge,4),
                'cosine': round(cosine,4),
            })

            if (j+1) % 25 == 0:
                print(f'  [{j+1}/{n_sample}]  '
                      f'BLEU={np.mean(bleus):.3f}  '
                      f'ROUGE-L={np.mean(rouges):.3f}  '
                      f'Cosine={np.mean(cosines):.3f}')

        print(f'  Fold {fold} final — '
              f'BLEU={np.mean(bleus):.4f}  '
              f'ROUGE-L={np.mean(rouges):.4f}  '
              f'Cosine={np.mean(cosines):.4f}  '
              f'({time.time()-t0:.0f}s)')

    save_results(all_rows, f'{RESULTS_DIR}/p4b_results.csv')

    # ── Summary ───────────────────────────────────────────────────────────
    res_df = pd.DataFrame(all_rows)
    sep('TASK B SUMMARY')
    print(f"  BLEU    : {res_df['bleu'].mean():.4f}")
    print(f"  ROUGE-L : {res_df['rouge_l'].mean():.4f}")
    print(f"  Cosine  : {res_df['cosine'].mean():.4f}")
    print(f"  N eval  : {len(res_df)}")
    print('\n  [OK] Stage P4B complete.')

# =============================================================================
# STAGE P5 -- ABLATION STUDY  (fold 0 only)
# =============================================================================

def _p5_done():
    return os.path.exists(f'{RESULTS_DIR}/p5_ablation.csv')

def run_p5():
    if _p5_done():
        print('  [>>] Stage P5 already done. Skipping.')
        return
    sep('STAGE P5 -- ABLATION STUDY  (fold 0)')
    import lightgbm as lgb
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as sk_normalize
    import faiss as faiss_lib
    import ollama
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer as rouge_module
    from sentence_transformers import SentenceTransformer

    with open(f'{FAISS_DIR}/fold0_meta.pkl', 'rb') as fh:
        m = pickle.load(fh)

    X_tr, X_va   = m['X_tr'], m['X_va']
    y_tr, y_va   = m['y_A_tr'], m['y_A_va']
    bert_tr      = m['bert_tr']
    bert_va      = m['bert_va']
    raw_tr       = m['raw_tr']
    raw_va       = m['raw_va']
    n_cls        = int(y_tr.max()) + 1
    results      = []

    def _lgb_train_eval(Xtr, Xva, label=''):
        params = {**LGB_PARAMS, 'num_class': n_cls}
        ds_tr  = lgb.Dataset(Xtr, label=y_tr)
        ds_va  = lgb.Dataset(Xva, label=y_va, reference=ds_tr)
        bst    = lgb.train(params, ds_tr,
                           num_boost_round=LGB_ROUNDS,
                           valid_sets=[ds_va],
                           callbacks=[lgb.early_stopping(LGB_EARLY,verbose=False),
                                      lgb.log_evaluation(200)])
        proba  = bst.predict(Xva)
        top1   = float(np.mean(proba.argmax(1) == y_va))
        top3   = topk_accuracy(y_va, proba, 3)
        top5   = topk_accuracy(y_va, proba, 5)
        print(f'  {label:30s} top1={top1:.4f} top3={top3:.4f} top5={top5:.4f}')
        return top1, top3, top5

    # A1 -- BERT only
    t1, t3, t5 = _lgb_train_eval(bert_tr, bert_va, 'BERT only')
    results.append({'variant':'BERT only','top1':t1,'top3':t3,'top5':t5})

    # A2 -- Tabular only (no BERT)
    X_tab_tr = X_tr[:, 768:]
    X_tab_va = X_va[:, 768:]
    t1, t3, t5 = _lgb_train_eval(X_tab_tr, X_tab_va, 'Tabular only')
    results.append({'variant':'Tabular only','top1':t1,'top3':t3,'top5':t5})

    # A3 -- BERT + Tabular (full, same as P4A)
    t1, t3, t5 = _lgb_train_eval(X_tr, X_va, 'BERT + Tabular (full)')
    results.append({'variant':'BERT + Tabular','top1':t1,'top3':t3,'top5':t5})

    # A4 -- TF-IDF baseline (scheduled_procedure text only)
    tfidf  = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    tfidf_tr = tfidf.fit_transform(raw_tr[TEXT_COL].fillna('').tolist()).toarray()
    tfidf_va = tfidf.transform(raw_va[TEXT_COL].fillna('').tolist()).toarray()
    t1, t3, t5 = _lgb_train_eval(tfidf_tr.astype(np.float32),
                                  tfidf_va.astype(np.float32), 'TF-IDF')
    results.append({'variant':'TF-IDF','top1':t1,'top3':t3,'top5':t5})

    # B -- RAG K ablation
    scorer_rouge = rouge_module.RougeScorer(['rougeL'], use_stemmer=False)
    st_model     = SentenceTransformer('all-MiniLM-L6-v2')
    smoother     = SmoothingFunction().method1
    index        = faiss_lib.read_index(f'{FAISS_DIR}/fold0_index.faiss')
    bert_va_f32  = bert_va.astype(np.float32)

    rng        = np.random.default_rng(RANDOM_STATE)
    n_sample   = min(50, len(raw_va))
    sample_idx = rng.choice(len(raw_va), n_sample, replace=False)

    for k in [1, 3, 5, 10]:
        bleus, rouges, cosines = [], [], []
        for vi in sample_idx:
            row = raw_va.iloc[vi].to_dict()
            ref = str(row[TARGET_B])
            if not ref or ref == 'nan': continue
            q      = bert_va_f32[vi:vi+1]
            q_norm = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-8)
            _, nn  = index.search(q_norm, k)
            nbs    = [{'proc': raw_tr.iloc[ni][TEXT_COL],
                       'opdx': raw_tr.iloc[ni][TARGET_B]}
                      for ni in nn[0] if ni < len(raw_tr)]
            prompt = _rag_prompt(row, nbs)
            try:
                resp = ollama.generate(model=OLLAMA_MODEL, prompt=prompt,
                                       options={'num_predict':30,'temperature':0.1})
                pred = resp['response'].strip().split('\n')[0].lower()
            except: pred = ''
            if not pred: continue
            bleus.append(sentence_bleu([ref.split()], pred.split(), smoothing_function=smoother))
            rouges.append(scorer_rouge.score(ref, pred)['rougeL'].fmeasure)
            e = st_model.encode([ref, pred], normalize_embeddings=True)
            cosines.append(float(np.dot(e[0], e[1])))

        row_r = {'variant': f'RAG K={k}',
                 'bleu':    round(np.mean(bleus),4)   if bleus   else 0,
                 'rouge_l': round(np.mean(rouges),4)  if rouges  else 0,
                 'cosine':  round(np.mean(cosines),4) if cosines else 0}
        print(f"  RAG K={k:2d}  BLEU={row_r['bleu']:.4f}  "
              f"ROUGE-L={row_r['rouge_l']:.4f}  "
              f"Cosine={row_r['cosine']:.4f}")
        results.append(row_r)

    save_results(results, f'{RESULTS_DIR}/p5_ablation.csv')
    sep('ABLATION SUMMARY')
    print(pd.DataFrame(results).to_string(index=False))
    print('\n  [OK] Stage P5 complete.')

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    sep('PAPER PIPELINE START')
    print(f'  Output dir : {RESULTS_DIR}')
    print(f'  FAISS dir  : {FAISS_DIR}')

    run_p1()
    run_p2()
    run_p3()
    run_p4a()
    run_p4b()
    run_p5()

    sep('ALL STAGES COMPLETE')
    print('  Results in:', RESULTS_DIR)
