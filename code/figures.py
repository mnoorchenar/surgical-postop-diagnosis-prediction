"""
make_figures.py  --  generate all paper figures + updated flowchart
Outputs (all in paper/figures/):
  fig1_methodology.pdf   -- graphviz flowchart
  fig2_task_a.pdf        -- Task A bar chart (top-1/3/5)
  fig3_ablation_a.pdf    -- Ablation ICD variants
  fig4_rag_k.pdf         -- RAG K ablation line chart
  fig5_task_b_dist.pdf   -- Task B metric distributions (box plot)
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import graphviz

os.makedirs('paper/figures', exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────
C1 = '#2471A3'   # dark blue   – baseline / BERT
C2 = '#1E8449'   # dark green  – SGD / BERT+Tab
C3 = '#884EA0'   # purple      – LightGBM
C4 = '#CB4335'   # red         – TF-IDF
C5 = '#D4AC0D'   # gold        – tabular only
GREY = '#5D6D7E'

plt.rcParams.update({
    'font.family':  'DejaVu Sans',
    'font.size':    11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi':   150,
})

# ==========================================================================
# FIG 1 -- METHODOLOGY FLOWCHART (graphviz)
# ==========================================================================
def make_flowchart():
    g = graphviz.Digraph(
        name='Methodology',
        format='pdf',
        graph_attr={
            'rankdir':  'TB',
            'splines':  'polyline',
            'nodesep':  '0.65',
            'ranksep':  '0.70',
            'fontname': 'Helvetica',
            'bgcolor':  'white',
            'pad':      '0.45',
            'dpi':      '300',
        },
        node_attr={'fontname':'Helvetica','fontsize':'13','penwidth':'1.8','margin':'0.2,0.12'},
        edge_attr={'fontname':'Helvetica','fontsize':'11','penwidth':'1.6','color':'#2C3E50'},
    )
    BDR = '#2C3E50'
    def nd(g, n, lbl, fill, shape='box', style='filled,rounded', fc='#1A1A1A'):
        g.node(n, label=lbl, fillcolor=fill, style=style, shape=shape, color=BDR, fontcolor=fc)

    # title
    g.node('title',
           label='Pre-operative Text-Driven Prediction of Post-operative Diagnoses',
           shape='plaintext', fontname='Helvetica-Bold', fontsize='14', fontcolor='#1A252F')

    # dataset
    nd(g,'data','194,661 Surgical Cases\ncasetime.csv  |  13 Services  |  10+ Years',
       '#D6EAF8', shape='cylinder')

    # pre-op inputs
    with g.subgraph(name='cluster_preop') as c:
        c.attr(label='Pre-operative Inputs  (strictly available before surgery)',
               style='filled', fillcolor='#EBF5FB', color='#2471A3',
               fontname='Helvetica-Bold', fontsize='12', penwidth='2.5')
        nd(c,'text_in','Text Input\nscheduled_procedure\n1,810 unique  |  avg 25 chars','#AED6F1')
        nd(c,'struct_in','Structured Input\nage  BMI  ASA  sex  OR team size\nservice  anesthetic  scheduled duration\nlocation  start hour  day/month','#AED6F1')

    # encoding
    with g.subgraph(name='cluster_enc') as c:
        c.attr(rank='same', style='invis')
        nd(c,'clinbert','Bio_ClinicalBERT\n768-d CLS embedding','#FDEBD0')
        nd(c,'tab_enc','Tabular Encoder\none-hot + median impute\n(fold-wise, no leakage)','#FDEBD0')
        nd(c,'faiss','FAISS Index\n(fold-isolated)\n147K train vectors','#FDEBD0',shape='cylinder')

    # fusion
    nd(g,'fuse','Feature Fusion\nBERT embed  ||  Tabular embed\n806-dimensional shared representation','#F9EBEA',shape='diamond')

    # CV
    nd(g,'cv','5-Fold Stratified Cross-Validation','#F2F3F4')

    # task A
    with g.subgraph(name='cluster_A') as c:
        c.attr(label='Task A  —  ICD-10 Classification',
               style='filled', fillcolor='#EAF2FF', color='#1A5276',
               fontname='Helvetica-Bold', fontsize='12', penwidth='2.5')
        nd(c,'sgd','SGD Classifier\nL2-norm BERT features','#AED6F1')
        nd(c,'lgb','LightGBM\nBERT + Tabular  |  557 classes','#AED6F1')
        nd(c,'out_a','most_responsible_dx\nICD-10 diagnosis code + text\nBaseline top-1: 53.8%','#2471A3',fc='white')
        nd(c,'eval_a','Top-1 / Top-3 / Top-5 Accuracy\nMacro-F1','#D6EAF8')

    # task B
    with g.subgraph(name='cluster_B') as c:
        c.attr(label='Task B  —  Operative Diagnosis Generation',
               style='filled', fillcolor='#EAFAF1', color='#1E8449',
               fontname='Helvetica-Bold', fontsize='12', penwidth='2.5')
        nd(c,'rag','FAISS  top-K retrieval\n+  Llama-3  (Ollama, local)\nRAG prompt generation','#A9DFBF')
        nd(c,'out_b','operative_dx\nFree-text clinical finding\nBaseline top-1: 29.4%','#1E8449',fc='white')
        nd(c,'eval_b','BLEU  |  ROUGE-L\nCosine Similarity (SentenceBERT)','#D5F5E3')

    # ablation
    nd(g,'ablation',
       'Ablation Study\nText-only  vs  Text+Structured  vs  TF-IDF\nRAG K = 1 / 3 / 5 / 10',
       '#D7BDE2')

    # edges
    g.edge('title',    'data',      style='invis')
    g.edge('data',     'text_in')
    g.edge('data',     'struct_in')
    g.edge('text_in',  'clinbert')
    g.edge('struct_in','tab_enc')
    g.edge('clinbert', 'faiss',  style='dashed', label='index')
    g.edge('clinbert', 'fuse')
    g.edge('tab_enc',  'fuse')
    g.edge('fuse',     'cv')
    g.edge('cv',       'sgd')
    g.edge('cv',       'lgb')
    g.edge('cv',       'rag')
    g.edge('faiss',    'rag',    label='retrieve K')
    g.edge('sgd',      'out_a')
    g.edge('lgb',      'out_a')
    g.edge('rag',      'out_b')
    g.edge('out_a',    'eval_a')
    g.edge('out_b',    'eval_b')
    g.edge('eval_a',   'ablation', style='dashed')
    g.edge('eval_b',   'ablation', style='dashed')

    out = g.render(filename='paper/figures/fig1_methodology', cleanup=True, view=False)
    print(f'  Saved: {out}')

make_flowchart()

# ==========================================================================
# FIG 2 -- TASK A: Top-K Accuracy bar chart
# ==========================================================================
def make_fig2():
    models  = ['Baseline\n(Mode/Procedure)', 'SGD\n(BERT)', 'LightGBM\n(BERT+Tabular)']
    top1    = [0.5376, 0.5037, 0.3379]
    top3    = [0.5376, 0.7373, 0.6510]
    top5    = [0.5376, 0.8129, 0.7761]
    x       = np.arange(len(models))
    w       = 0.26

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w,   top1, w, label='Top-1', color=C1, edgecolor='white', linewidth=0.5)
    b3 = ax.bar(x,       top3, w, label='Top-3', color=C2, edgecolor='white', linewidth=0.5)
    b5 = ax.bar(x + w,   top5, w, label='Top-5', color=C3, edgecolor='white', linewidth=0.5)

    for bars in [b1, b3, b5]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax.set_ylabel('Accuracy')
    ax.set_title('Task A: ICD-10 Classification Accuracy (5-fold CV)\n557 classes, 183,913 cases')
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylim(0, 0.97)
    ax.legend(loc='upper right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.axhline(0.538, color=GREY, linestyle='--', linewidth=1.0, alpha=0.6, label='_')
    ax.text(2.55, 0.545, 'Baseline\ntop-1', fontsize=8.5, color=GREY, ha='right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('paper/figures/fig2_task_a.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved: paper/figures/fig2_task_a.pdf')

make_fig2()

# ==========================================================================
# FIG 3 -- ABLATION: ICD classification feature variants (fold 0)
# ==========================================================================
def make_fig3():
    variants = ['Tabular\nOnly', 'BERT\nOnly', 'BERT +\nTabular', 'TF-IDF']
    top1 = [0.1097, 0.3896, 0.3365, 0.4070]
    top3 = [0.3397, 0.6583, 0.6493, 0.6690]
    top5 = [0.4988, 0.7825, 0.7739, 0.7747]
    colors = [C5, C1, C2, C4]
    x = np.arange(len(variants)); w = 0.26

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w, top1, w, label='Top-1', color=C1, edgecolor='white')
    b3 = ax.bar(x,     top3, w, label='Top-3', color=C2, edgecolor='white')
    b5 = ax.bar(x + w, top5, w, label='Top-5', color=C3, edgecolor='white')
    for bars in [b1, b3, b5]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.008,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Accuracy (fold 0)')
    ax.set_title('Task A Ablation: Feature Variants for ICD Classification')
    ax.set_xticks(x); ax.set_xticklabels(variants)
    ax.set_ylim(0, 0.97)
    ax.legend(loc='upper left')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('paper/figures/fig3_ablation_a.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved: paper/figures/fig3_ablation_a.pdf')

make_fig3()

# ==========================================================================
# FIG 4 -- RAG K ablation line chart
# ==========================================================================
def make_fig4():
    ks      = [1, 3, 5, 10]
    bleu    = [0.1269, 0.1438, 0.1478, 0.1817]
    rouge   = [0.4055, 0.4541, 0.4943, 0.5175]
    cosine  = [0.5843, 0.6323, 0.6769, 0.6780]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, bleu,   'o-', color=C1, linewidth=2, markersize=7, label='BLEU')
    ax.plot(ks, rouge,  's-', color=C2, linewidth=2, markersize=7, label='ROUGE-L')
    ax.plot(ks, cosine, '^-', color=C3, linewidth=2, markersize=7, label='Cosine Similarity')

    for xi, (b, r, c) in zip(ks, zip(bleu, rouge, cosine)):
        ax.annotate(f'{b:.3f}', (xi, b), textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=9, color=C1)
        ax.annotate(f'{r:.3f}', (xi, r), textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=9, color=C2)
        ax.annotate(f'{c:.3f}', (xi, c), textcoords='offset points', xytext=(0,-14),
                    ha='center', fontsize=9, color=C3)

    ax.set_xlabel('Number of Retrieved Neighbours (K)')
    ax.set_ylabel('Score')
    ax.set_title('Task B Ablation: Effect of RAG Retrieval Size K\n(fold 0, n=50 test cases)')
    ax.set_xticks(ks)
    ax.set_ylim(0.05, 0.75)
    ax.legend(loc='lower right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('paper/figures/fig4_rag_k.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved: paper/figures/fig4_rag_k.pdf')

make_fig4()

# ==========================================================================
# FIG 5 -- Task B distribution box plot (BLEU / ROUGE-L / Cosine)
# ==========================================================================
def make_fig5():
    df = pd.read_csv('results/paper/p4b_results.csv')

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.5))
    metrics = [('bleu', 'BLEU', C1), ('rouge_l', 'ROUGE-L', C2), ('cosine', 'Cosine Similarity', C3)]

    for ax, (col, title, color) in zip(axes, metrics):
        data_by_fold = [df[df['fold']==f][col].values for f in range(5)]
        bp = ax.boxplot(data_by_fold, patch_artist=True, notch=False,
                        medianprops={'color':'white','linewidth':2})
        for patch in bp['boxes']:
            patch.set_facecolor(color); patch.set_alpha(0.75)
        for whisker in bp['whiskers']: whisker.set(color=GREY, linewidth=1.2)
        for cap in bp['caps']:         cap.set(color=GREY, linewidth=1.2)
        for flier in bp['fliers']:     flier.set(marker='o', color=GREY, alpha=0.3, markersize=3)
        mean_val = df[col].mean()
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
        ax.text(5.4, mean_val, f'  μ={mean_val:.3f}', va='center', fontsize=9.5, color='red')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Fold'); ax.set_ylabel('Score')
        ax.set_xticks(range(1, 6)); ax.set_xticklabels([f'F{i}' for i in range(5)])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.suptitle('Task B: RAG + Llama-3 Generation Metrics per Fold (n=100/fold)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('paper/figures/fig5_task_b_dist.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved: paper/figures/fig5_task_b_dist.pdf')

make_fig5()

# ==========================================================================
# FIG 6 -- Per-fold consistency (Task A top-3, Task B ROUGE-L)
# ==========================================================================
def make_fig6():
    df_a = pd.read_csv('results/paper/p4a_results.csv')
    df_b = pd.read_csv('results/paper/p4b_results.csv')

    folds = list(range(5))
    sgd_top3 = df_a[df_a['model']=='SGD(BERT)']['top3'].values
    lgb_top3 = df_a[df_a['model']=='LightGBM']['top3'].values
    rag_rouge = df_b.groupby('fold')['rouge_l'].mean().values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Task A per-fold
    ax1.plot(folds, sgd_top3, 'o-', color=C2, lw=2, ms=7, label='SGD(BERT) Top-3')
    ax1.plot(folds, lgb_top3, 's-', color=C3, lw=2, ms=7, label='LightGBM Top-3')
    ax1.axhline(np.mean(sgd_top3), color=C2, lw=1, ls='--', alpha=0.5)
    ax1.axhline(np.mean(lgb_top3), color=C3, lw=1, ls='--', alpha=0.5)
    ax1.set_xlabel('Fold'); ax1.set_ylabel('Top-3 Accuracy')
    ax1.set_title('Task A: Top-3 Accuracy per Fold')
    ax1.set_xticks(folds); ax1.set_xticklabels([f'F{i}' for i in folds])
    ax1.set_ylim(0.55, 0.80); ax1.legend()
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3)

    # Task B per-fold
    ax2.bar(folds, rag_rouge, color=C2, alpha=0.75, edgecolor='white')
    ax2.axhline(np.mean(rag_rouge), color='red', lw=1.5, ls='--', alpha=0.8,
                label=f'Mean={np.mean(rag_rouge):.3f}')
    for i, v in enumerate(rag_rouge):
        ax2.text(i, v+0.005, f'{v:.3f}', ha='center', fontsize=9.5, fontweight='bold')
    ax2.set_xlabel('Fold'); ax2.set_ylabel('ROUGE-L')
    ax2.set_title('Task B: ROUGE-L per Fold (n=100/fold)')
    ax2.set_xticks(folds); ax2.set_xticklabels([f'F{i}' for i in folds])
    ax2.set_ylim(0, 0.65); ax2.legend()
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('paper/figures/fig6_per_fold.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved: paper/figures/fig6_per_fold.pdf')

make_fig6()

print('\nAll figures done.')
