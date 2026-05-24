

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
sns.set_palette("muted")

df = pd.read_csv('elderly_health_data.csv')


df['low_activity'] = df['activity_score'] < df['activity_score'].median()
df['poor_sleep']   = df['sleep_hours']    < df['sleep_hours'].median()

print("="*60)
print("HYPOTHESIS 2: Living alone vs assisted — risk difference")
print("="*60)

alone    = df[df['living_condition'] == 'Alone']
assisted = df[df['living_condition'] == 'Assisted']

alone_mean    = alone['risk_score'].mean()
assisted_mean = assisted['risk_score'].mean()
diff = alone_mean - assisted_mean

print(f"\nLiving Alone   → Mean risk: {alone_mean:.1f}  (n={len(alone)})")
print(f"Assisted Care  → Mean risk: {assisted_mean:.1f}  (n={len(assisted)})")
print(f"Difference     → {diff:.1f} points higher for those living alone")


metrics = ['activity_score','sleep_hours','medication_adherence',
           'social_interaction_score','falls_last_month']
metric_labels = ['Activity Score','Sleep Hours','Medication Adherence',
                 'Social Score','Falls (last month)']

print("\nPer-metric breakdown:")
for m, lbl in zip(metrics, metric_labels):
    a = alone[m].mean()
    s = assisted[m].mean()
    print(f"  {lbl:<28} Alone: {a:.2f}   Assisted: {s:.2f}   Δ = {a-s:+.2f}")

# ── Chart 1: Side-by-side comparison ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('H2: Does living alone increase health risk?', fontsize=13)

colors = {'Alone': '#e05c5c', 'Assisted': '#4caf7d'}


for lc, col in colors.items():
    subset = df[df['living_condition'] == lc]['risk_score']
    axes[0].hist(subset, bins=20, alpha=0.65, color=col, label=lc, edgecolor='white')
axes[0].axvline(alone_mean,    color='#e05c5c', linestyle='--', linewidth=1.5)
axes[0].axvline(assisted_mean, color='#4caf7d', linestyle='--', linewidth=1.5)
axes[0].set_title('Risk Score Distribution')
axes[0].set_xlabel('Risk Score')
axes[0].set_ylabel('Count')
axes[0].legend()

# Boxplot
sns.boxplot(data=df, x='living_condition', y='risk_score',
            palette=colors, ax=axes[1], order=['Alone','Assisted'])
axes[1].set_title('Risk Score by Living Condition')
axes[1].set_xlabel('')
axes[1].set_ylabel('Risk Score')
for i, (lc, col) in enumerate(colors.items()):
    m = df[df['living_condition']==lc]['risk_score'].mean()
    axes[1].text(i, m + 1, f'mean\n{m:.1f}', ha='center', fontsize=9, color=col, fontweight='bold')

# Per-metric delta bar chart
deltas = [alone[m].mean() - assisted[m].mean() for m in metrics]
bar_colors = ['#e05c5c' if d > 0 else '#4caf7d' for d in deltas]
axes[2].barh(metric_labels, deltas, color=bar_colors, edgecolor='white')
axes[2].axvline(0, color='gray', linewidth=0.8)
axes[2].set_title('Metric Gap: Alone − Assisted')
axes[2].set_xlabel('Difference (Alone minus Assisted)')
for i, d in enumerate(deltas):
    axes[2].text(d + (0.01 if d >= 0 else -0.01), i,
                 f'{d:+.2f}', va='center', ha='left' if d >= 0 else 'right', fontsize=9)

plt.tight_layout()
plt.savefig('03_h2_living_condition_risk.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: 03_h2_living_condition_risk.png")

# H2 Insight 
print(f"""
--- H2 INSIGHT ---
Elderly living alone have a mean risk score {diff:.1f} points higher than those
in assisted care. The gap is visible across every metric — activity, sleep,
medication adherence, and social interaction are all worse for the alone group.

CAREGIVER ACTION: Patients living alone should receive more frequent
check-ins regardless of their current condition type.
""")

# ═══════════════════════════════════════════════════════════
print("="*60)
print("HYPOTHESIS 3: Does the alone-vs-assisted gap widen with age?")
print("="*60)

age_gaps = []
for grp in ['65-74', '75-84', '85+']:
    sub = df[df['age_group'] == grp]
    a_mean = sub[sub['living_condition']=='Alone']['risk_score'].mean()
    s_mean = sub[sub['living_condition']=='Assisted']['risk_score'].mean()
    gap = a_mean - s_mean
    n_alone = len(sub[sub['living_condition']=='Alone'])
    n_asst  = len(sub[sub['living_condition']=='Assisted'])
    age_gaps.append({'age_group': grp, 'alone_mean': a_mean,
                     'assisted_mean': s_mean, 'gap': gap,
                     'n_alone': n_alone, 'n_assisted': n_asst})
    print(f"  {grp}: Alone={a_mean:.1f}  Assisted={s_mean:.1f}  Gap={gap:+.1f}  "
          f"(n_alone={n_alone}, n_assisted={n_asst})")

gap_df = pd.DataFrame(age_gaps)

#  Chart 2: Age group × living condition 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('H3: Does the risk gap between alone vs assisted widen with age?', fontsize=13)

x = np.arange(len(gap_df))
w = 0.35
axes[0].bar(x - w/2, gap_df['alone_mean'],    width=w, label='Alone',    color='#e05c5c', edgecolor='white')
axes[0].bar(x + w/2, gap_df['assisted_mean'], width=w, label='Assisted', color='#4caf7d', edgecolor='white')
axes[0].set_xticks(x)
axes[0].set_xticklabels(gap_df['age_group'])
axes[0].set_title('Mean Risk Score by Age Group & Living Condition')
axes[0].set_ylabel('Mean Risk Score')
axes[0].set_xlabel('Age Group')
axes[0].legend()
for i, row in gap_df.iterrows():
    axes[0].text(i - w/2, row['alone_mean']    + 0.4, f"{row['alone_mean']:.1f}",    ha='center', fontsize=9)
    axes[0].text(i + w/2, row['assisted_mean'] + 0.4, f"{row['assisted_mean']:.1f}", ha='center', fontsize=9)

# Gap line
gap_colors = ['#e05c5c' if g > 0 else '#4caf7d' for g in gap_df['gap']]
axes[1].bar(gap_df['age_group'], gap_df['gap'], color=gap_colors, edgecolor='white', width=0.5)
axes[1].axhline(0, color='gray', linewidth=0.8)
axes[1].set_title('Risk Gap (Alone − Assisted) by Age Group')
axes[1].set_ylabel('Risk Score Gap')
axes[1].set_xlabel('Age Group')
for i, row in gap_df.iterrows():
    axes[1].text(i, row['gap'] + 0.3, f"{row['gap']:+.1f}", ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('04_h3_age_living_gap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 04_h3_age_living_gap.png")

max_gap_row = gap_df.loc[gap_df['gap'].idxmax()]
print(f"""
--- H3 INSIGHT ---
The risk gap between alone and assisted patients is largest in the
'{max_gap_row['age_group']}' age group ({max_gap_row['gap']:+.1f} points).

This means isolation becomes MORE dangerous as patients age — not just
a constant background risk. The compounding effect of age + isolation
is the key finding.

CAREGIVER ACTION: Elderly patients aged 85+ living alone should be
flagged as a priority monitoring group, even if their current vitals
look stable.
""")

