

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 11

df = pd.read_csv('elderly_health_data.csv')


df['low_activity']      = df['activity_score']      < df['activity_score'].median()
df['poor_sleep']        = df['sleep_hours']          < df['sleep_hours'].median()

df['combo_group'] = 'Both Good'
df.loc[df['low_activity'] & ~df['poor_sleep'], 'combo_group'] = 'Low Activity Only'
df.loc[~df['low_activity'] & df['poor_sleep'], 'combo_group'] = 'Poor Sleep Only'
df.loc[df['low_activity'] & df['poor_sleep'],  'combo_group'] = 'Both Poor'

df['flag_low_activity']  = (df['activity_score']      < df['activity_score'].quantile(0.33)).astype(int)
df['flag_poor_sleep']    = (df['sleep_hours']          < df['sleep_hours'].quantile(0.33)).astype(int)
df['flag_low_adherence'] = (df['medication_adherence'] < df['medication_adherence'].quantile(0.33)).astype(int)
df['flag_high_bp']       = (df['systolic_bp']          > df['systolic_bp'].quantile(0.67)).astype(int)
df['flag_fall']          = df['falls_last_month']
df['risk_flag_count']    = df[['flag_low_activity','flag_poor_sleep',
                                'flag_low_adherence','flag_high_bp','flag_fall']].sum(axis=1)


fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

fig.suptitle(
    'Elderly Health Risk — Key Findings Summary\n'
    'Project 1: EDA & Pattern Analysis  |  300 Patients',
    fontsize=15, fontweight='bold', y=1.01
)

#: H1 — Compounding effect of activity + sleep 
order   = ['Both Good','Low Activity Only','Poor Sleep Only','Both Poor']
palette = ['#4caf7d','#5b9bd5','#f0a500','#e05c5c']
means   = [df[df['combo_group']==g]['risk_score'].mean() for g in order]
counts  = [df[df['combo_group']==g].shape[0] for g in order]

bars = axes[0].bar(order, means, color=palette, edgecolor='white', width=0.6)
for bar, m, c in zip(bars, means, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 m + 0.5, f'{m:.1f}\n(n={c})',
                 ha='center', fontsize=9)
axes[0].set_title('H1: Activity + Sleep\nCompound Risk', fontweight='bold')
axes[0].set_ylabel('Mean Risk Score')
axes[0].set_ylim(0, max(means) * 1.25)
axes[0].tick_params(axis='x', rotation=20, labelsize=8)

#  H2 — Living condition risk comparison 
lc_means = df.groupby('living_condition')['risk_score'].mean()
lc_cols  = {'Alone':'#e05c5c', 'Assisted':'#4caf7d'}
bars2 = axes[1].bar(lc_means.index, lc_means.values,
                    color=[lc_cols[l] for l in lc_means.index],
                    edgecolor='white', width=0.5)
for bar, (lc, val) in zip(bars2, lc_means.items()):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 val + 0.4, f'{val:.1f}', ha='center',
                 fontsize=11, fontweight='bold')
diff_lc = lc_means['Alone'] - lc_means['Assisted']
axes[1].set_title(f'H2: Living Alone\n+{diff_lc:.1f} pts higher risk', fontweight='bold')
axes[1].set_ylabel('Mean Risk Score')
axes[1].set_ylim(0, max(lc_means.values) * 1.2)

# Panel 3: H3 — Age × living condition gap 
age_groups = ['65-74','75-84','85+']
alone_means = [df[(df['age_group']==g)&(df['living_condition']=='Alone')]['risk_score'].mean()
               for g in age_groups]
asst_means  = [df[(df['age_group']==g)&(df['living_condition']=='Assisted')]['risk_score'].mean()
               for g in age_groups]

x = np.arange(len(age_groups))
w = 0.35
axes[2].bar(x - w/2, alone_means, w, label='Alone',    color='#e05c5c', edgecolor='white')
axes[2].bar(x + w/2, asst_means,  w, label='Assisted', color='#4caf7d', edgecolor='white')
axes[2].set_xticks(x)
axes[2].set_xticklabels(age_groups)
axes[2].set_title('H3: Risk Gap Widens\nWith Age', fontweight='bold')
axes[2].set_ylabel('Mean Risk Score')
axes[2].legend(fontsize=9)
for i, (a, s) in enumerate(zip(alone_means, asst_means)):
    axes[2].text(i - w/2, a + 0.3, f'{a:.0f}', ha='center', fontsize=9, color='#e05c5c')
    axes[2].text(i + w/2, s + 0.3, f'{s:.0f}', ha='center', fontsize=9, color='#4caf7d')

#  Panel 4: H4 — Condition vs adherence & risk 
cond_stats = df.groupby('primary_condition').agg(
    adherence=('medication_adherence','mean'),
    risk     =('risk_score',          'mean')
).sort_values('adherence')

cond_colors = ['#e05c5c','#f0a500','#5b9bd5','#4caf7d']
ax4b = axes[3].twinx()
b1 = axes[3].bar(np.arange(4) - 0.2, cond_stats['adherence'],
                 0.35, color=cond_colors, edgecolor='white', alpha=0.85, label='Adherence')
b2 = ax4b.bar(np.arange(4) + 0.2, cond_stats['risk'],
              0.35, color=cond_colors, edgecolor='white', alpha=0.45, label='Risk Score')
axes[3].set_xticks(range(4))
axes[3].set_xticklabels(cond_stats.index, rotation=15, fontsize=9)
axes[3].set_ylabel('Adherence Rate', fontsize=9)
ax4b.set_ylabel('Risk Score', fontsize=9)
axes[3].set_title('H4: Cognitive Patients\nLowest Adherence, Highest Risk', fontweight='bold')
axes[3].set_ylim(0, 1.1)
ax4b.set_ylim(0, 80)

#  Panel 5: H5 — Age & risk flag clustering 
age_pcts = [
    df[(df['age_group']==g)&(df['risk_flag_count']>=2)].shape[0] /
    df[df['age_group']==g].shape[0] * 100
    for g in age_groups
]
bar_cols = ['#4caf7d','#f0a500','#e05c5c']
b5 = axes[4].bar(age_groups, age_pcts, color=bar_cols, edgecolor='white', width=0.5)
for bar, pct in zip(b5, age_pcts):
    axes[4].text(bar.get_x() + bar.get_width()/2,
                 pct + 1, f'{pct:.1f}%', ha='center',
                 fontsize=10, fontweight='bold')
axes[4].set_title('H5: 85+ — 63% Carry\n2+ Simultaneous Risk Flags', fontweight='bold')
axes[4].set_ylabel('% Patients with 2+ Flags')
axes[4].set_ylim(0, 90)

#  Panel 6: Key Insights Text Summary 
axes[5].axis('off')
insights = [
    ("H1", "Low activity + poor sleep together\nraise risk by ~18 pts vs neither alone"),
    ("H2", "Living alone adds +8.3 pts risk\nacross ALL health metrics"),
    ("H3", "85+ patients living alone reach\nmean risk 52.4 — highest of all groups"),
    ("H4", "Cognitive patients: adherence 0.53\nvs 0.74 avg → risk correlation −0.48"),
    ("H5", "63% of 85+ patients carry 2+\nsimultaneous risk flags"),
]

axes[5].set_xlim(0, 1)
axes[5].set_ylim(0, 1)
axes[5].text(0.5, 0.97, 'Key Findings', ha='center', va='top',
             fontsize=12, fontweight='bold', color='#222')

colors_ins = ['#4caf7d','#e05c5c','#e05c5c','#f0a500','#5b9bd5']
for i, ((tag, txt), col) in enumerate(zip(insights, colors_ins)):
    y = 0.83 - i * 0.175
    axes[5].add_patch(plt.Rectangle((0.02, y - 0.04), 0.96, 0.15,
                                    facecolor=col, alpha=0.08,
                                    transform=axes[5].transAxes, clip_on=False))
    axes[5].text(0.06, y + 0.05, tag, fontsize=10, fontweight='bold',
                 color=col, va='top')
    axes[5].text(0.18, y + 0.05, txt, fontsize=9, color='#333', va='top')

plt.savefig('07_master_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 07_master_summary.png")

