import graphviz

g = graphviz.Digraph(
    name='SurgicalPrediction',
    format='pdf',
    graph_attr={
        'rankdir':  'TB',
        'splines':  'polyline',
        'nodesep':  '0.7',
        'ranksep':  '0.65',
        'fontname': 'Helvetica',
        'bgcolor':  'white',
        'pad':      '0.4',
        'dpi':      '300',
    },
    node_attr={
        'fontname': 'Helvetica',
        'fontsize': '13',
        'penwidth': '1.8',
        'margin':   '0.18,0.12',
    },
    edge_attr={
        'fontname': 'Helvetica',
        'fontsize': '11',
        'penwidth': '1.6',
        'color':    '#2C3E50',
    },
)

BDR = '#2C3E50'

def nd(g, n, lbl, fill, shape='box', style='filled,rounded', fontcolor='#1A1A1A'):
    g.node(n, label=lbl, fillcolor=fill, style=style,
           shape=shape, color=BDR, fontcolor=fontcolor)

# ── TITLE ────────────────────────────────────────────────────────────────────
g.node('title',
    label='Pre-operative Text-Driven Prediction of Post-operative Diagnoses',
    shape='plaintext', fontname='Helvetica-Bold', fontsize='15',
    fontcolor='#1A252F')

# ── DATASET ──────────────────────────────────────────────────────────────────
nd(g, 'data',
   '194,661 Surgical Cases\ncasetime.csv  |  13 Services  |  10+ Years',
   '#D6EAF8', shape='cylinder')

# ── PRE-OP INPUTS (side by side) ─────────────────────────────────────────────
with g.subgraph(name='cluster_preop') as c:
    c.attr(label='Pre-operative Inputs  (available before surgery)',
           style='filled', fillcolor='#EBF5FB', color='#2471A3',
           fontname='Helvetica-Bold', fontsize='13', penwidth='2.5', rank='same')

    nd(c, 'text_in',
       'Text Input\nscheduled_procedure\n1,810 unique  |  avg 25 chars',
       '#AED6F1')

    nd(c, 'struct_in',
       'Structured Input\nage  BMI  ASA  sex\nservice  anesthetic  team\nlocation  scheduled_duration',
       '#AED6F1')

# ── ENCODING ─────────────────────────────────────────────────────────────────
with g.subgraph(name='cluster_enc') as c:
    c.attr(rank='same', style='invis')

    nd(c, 'clinbert',
       'Bio_ClinicalBERT\n384-d embedding',
       '#FDEBD0')

    nd(c, 'tab_enc',
       'Tabular MLP\ndense embedding',
       '#FDEBD0')

    nd(c, 'faiss',
       'FAISS Index\n194K vectors\n(fold-isolated)',
       '#FDEBD0', shape='cylinder')

# ── FUSION ───────────────────────────────────────────────────────────────────
nd(g, 'fuse',
   'Feature Fusion\nShared Representation',
   '#F9EBEA', shape='diamond')

# ── 5-FOLD CV ────────────────────────────────────────────────────────────────
nd(g, 'cv',
   '5-Fold Cross-Validation\n(stratified  |  fold-wise encoding)',
   '#F2F3F4')

# ── TWO TASK BRANCHES ────────────────────────────────────────────────────────
with g.subgraph(name='cluster_taskA') as c:
    c.attr(label='Task A  —  ICD-10 Classification',
           style='filled', fillcolor='#EAF2FF', color='#1A5276',
           fontname='Helvetica-Bold', fontsize='13', penwidth='2.5')

    nd(c, 'cls_model',
       'ClinicalBERT + Classifier\nXGBoost  |  LightGBM  |  MLP\nLoRA fine-tune',
       '#AED6F1')

    nd(c, 'cls_out',
       'most_responsible_dx\n4,229 ICD-10 classes\nBaseline top-1: 49.3%',
       '#2471A3', fontcolor='white')

    nd(c, 'eval_a',
       'Top-1 / 3 / 5  Accuracy\nMacro-F1  |  AUROC',
       '#D6EAF8')

with g.subgraph(name='cluster_taskB') as c:
    c.attr(label='Task B  —  Operative Diagnosis Generation',
           style='filled', fillcolor='#EAFAF1', color='#1E8449',
           fontname='Helvetica-Bold', fontsize='13', penwidth='2.5')

    nd(c, 'rag_model',
       'FAISS Retrieval  top-K\n+  Llama-3  (Ollama local)\nRAG prompt generation',
       '#A9DFBF')

    nd(c, 'rag_out',
       'operative_dx\nFree-text clinical finding\nBaseline top-1: 29.4%',
       '#1E8449', fontcolor='white')

    nd(c, 'eval_b',
       'BLEU  |  ROUGE-L\nBERTScore  (ClinicalBERT)',
       '#D5F5E3')

# ── ABLATION ─────────────────────────────────────────────────────────────────
nd(g, 'ablation',
   'Ablation Study\nText-only  vs  Text + Structured\nClinicalBERT  vs  SentenceBERT  vs  TF-IDF\nRAG  k = 1 / 3 / 5 / 10',
   '#D7BDE2')

# ── EDGES ────────────────────────────────────────────────────────────────────
g.edge('title',      'data',       style='invis')
g.edge('data',       'text_in')
g.edge('data',       'struct_in')

g.edge('text_in',    'clinbert')
g.edge('struct_in',  'tab_enc')
g.edge('clinbert',   'faiss',    style='dashed', label='index')

g.edge('clinbert',   'fuse')
g.edge('tab_enc',    'fuse')

g.edge('fuse',       'cv')

g.edge('cv',         'cls_model')
g.edge('cv',         'rag_model')
g.edge('faiss',      'rag_model', label='retrieve top-K')

g.edge('cls_model',  'cls_out')
g.edge('rag_model',  'rag_out')

g.edge('cls_out',    'eval_a')
g.edge('rag_out',    'eval_b')

g.edge('eval_a',     'ablation',  style='dashed')
g.edge('eval_b',     'ablation',  style='dashed')

# ── RENDER ───────────────────────────────────────────────────────────────────
out = g.render(
    filename='flowchart',
    directory='D:/SurgeryProject/surgical-postop-diagnosis-prediction',
    cleanup=True,
    view=False,
)
print(f'Saved: {out}')
