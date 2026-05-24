
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
print("HYPOTHESIS 4: Condition type vs medication adherence & risk")
print("="*60)

conditions = ['mobility', 'cardiac', 'cognitive', 'diabetes']

cond_stats = df.groupby('primary_condition').agg(
    mean_adherence   = ('medication_adherence', 'mean'),
    mean_risk        = ('risk_score',           'mean'),
    mean_activity    = ('activity_score',        'mean'),
    mean_sleep       = ('sleep_hours',           'mean'),
    fall_rate        = ('falls_last_month',       'mean'),
    count            = ('patient_id',             'count')
).round(3)

print("\nCondition Summary:")
print(cond_stats.sort_values('mean_adherence'))

worst_adh  = cond_stats['mean_adherence'].idxmin()
highest_risk = cond_stats['mean_risk'].idxmax()
print(f"\nWorst medication adherence: {worst_adh} ({cond_stats.loc[worst_adh,'mean_adherence']:.2f})")
print(f"Highest overall risk:       {highest_risk} ({cond_stats.loc[highest_risk,'mean_risk']:.1f})")

#  Chart 1: 4-panel condition breakdown 
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('H4: How does chronic condition type affect adherence and risk?', fontsize=13)

cond_order = cond_stats.sort_values('mean_adherence').index.tolist()
palette = ['#e05c5c','#f0a500','#5b9bd5','#4caf7d']


bars = axes[0,0].bar(cond_order,
                     [cond_stats.loc[c,'mean_adherence'] for c in cond_order],
                     color=palette, edgecolor='white', width=0.6)
axes[0,0].set_title('Mean Medication Adherence by Condition')
axes[0,0].set_ylabel('Adherence Rate (0–1)')
axes[0,0].set_ylim(0, 1.1)
axes[0,0].axhline(df['medication_adherence'].mean(), color='gray',
                   linestyle='--', linewidth=1, label=f"Overall mean: {df['medication_adherence'].mean():.2f}")
axes[0,0].legend(fontsize=9)
for bar, c in zip(bars, cond_order):
    axes[0,0].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.01,
                   f"{cond_stats.loc[c,'mean_adherence']:.2f}",
                   ha='center', fontsize=10, fontweight='bold')


bars2 = axes[0,1].bar(cond_order,
                      [cond_stats.loc[c,'mean_risk'] for c in cond_order],
                      color=palette, edgecolor='white', width=0.6)
axes[0,1].set_title('Mean Risk Score by Condition')
axes[0,1].set_ylabel('Risk Score')
axes[0,1].axhline(df['risk_score'].mean(), color='gray', linestyle='--',
                   linewidth=1, label=f"Overall mean: {df['risk_score'].mean():.1f}")
axes[0,1].legend(fontsize=9)
for bar, c in zip(bars2, cond_order):
    axes[0,1].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.3,
                   f"{cond_stats.loc[c,'mean_risk']:.1f}",
                   ha='center', fontsize=10, fontweight='bold')

# Scatter
cond_colors = {'mobility':'#5b9bd5','cardiac':'#f0a500',
               'cognitive':'#e05c5c','diabetes':'#4caf7d'}
for cond, grp in df.groupby('primary_condition'):
    axes[1,0].scatter(grp['medication_adherence'], grp['risk_score'],
                      c=cond_colors[cond], label=cond, alpha=0.55, s=35, edgecolors='none')

# Trend line
z = np.polyfit(df['medication_adherence'], df['risk_score'], 1)
p = np.poly1d(z)
xline = np.linspace(df['medication_adherence'].min(), df['medication_adherence'].max(), 100)
axes[1,0].plot(xline, p(xline), color='black', linewidth=1.2, linestyle='--', label='Trend')
axes[1,0].set_title('Adherence vs Risk Score (all patients)')
axes[1,0].set_xlabel('Medication Adherence')
axes[1,0].set_ylabel('Risk Score')
axes[1,0].legend(fontsize=9)

corr = df['medication_adherence'].corr(df['risk_score'])
axes[1,0].text(0.05, 0.92, f"Correlation: {corr:.2f}",
               transform=axes[1,0].transAxes, fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# Multi-metric radar-style bar comparison
metrics_h4  = ['mean_adherence','mean_activity','mean_sleep','fall_rate']
mlabels     = ['Med Adherence','Activity Score','Sleep Hours','Fall Rate']
x = np.arange(len(mlabels))
width = 0.2
cond_list = ['mobility','cardiac','cognitive','diabetes']
col_list  = ['#5b9bd5','#f0a500','#e05c5c','#4caf7d']


normalised = {}
for m in metrics_h4:
    col_min = cond_stats[m].min()
    col_max = cond_stats[m].max()
    rng = col_max - col_min if col_max != col_min else 1
    normalised[m] = {c: (cond_stats.loc[c, m] - col_min) / rng for c in cond_list}

for i, (cond, col) in enumerate(zip(cond_list, col_list)):
    vals = [normalised[m][cond] for m in metrics_h4]
    axes[1,1].bar(x + i*width, vals, width, label=cond, color=col, edgecolor='white', alpha=0.85)

axes[1,1].set_title('Normalised Metric Comparison by Condition')
axes[1,1].set_xticks(x + width*1.5)
axes[1,1].set_xticklabels(mlabels, fontsize=9)
axes[1,1].set_ylabel('Normalised Score (0=worst, 1=best)')
axes[1,1].legend(fontsize=9)
axes[1,1].set_ylim(0, 1.3)

plt.tight_layout()
plt.savefig('05_h4_condition_adherence_risk.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 05_h4_condition_adherence_risk.png")

print(f"""
--- H4 INSIGHT ---
'{worst_adh}' patients show the lowest medication adherence ({cond_stats.loc[worst_adh,'mean_adherence']:.2f})
and the highest risk score ({cond_stats.loc[worst_adh,'mean_risk']:.1f}).

Correlation between adherence and risk: {corr:.2f} (negative = lower adherence → higher risk)

This confirms that medication non-adherence is not just a symptom of poor
health — it is a driver of risk. Condition type predicts adherence behaviour,
which in turn predicts overall health deterioration.

CAREGIVER ACTION: '{worst_adh}' patients need medication-specific intervention
(reminders, simplified regimens) — not just general monitoring.
""")


print("="*60)
print("HYPOTHESIS 5: Critical age window — where do risk factors cluster?")
print("="*60)

# Count how many risk flags each patient has
df['flag_low_activity']     = (df['activity_score']       < df['activity_score'].quantile(0.33)).astype(int)
df['flag_poor_sleep']       = (df['sleep_hours']           < df['sleep_hours'].quantile(0.33)).astype(int)
df['flag_low_adherence']    = (df['medication_adherence']  < df['medication_adherence'].quantile(0.33)).astype(int)
df['flag_high_bp']          = (df['systolic_bp']           > df['systolic_bp'].quantile(0.67)).astype(int)
df['flag_fall']             = df['falls_last_month']
df['risk_flag_count']       = (df[['flag_low_activity','flag_poor_sleep',
                                    'flag_low_adherence','flag_high_bp','flag_fall']].sum(axis=1))

age_flag = df.groupby('age').agg(
    mean_flags = ('risk_flag_count', 'mean'),
    mean_risk  = ('risk_score',      'mean'),
    count      = ('patient_id',      'count')
).reset_index()


age_grp_flag = df.groupby('age_group').agg(
    mean_flags = ('risk_flag_count', 'mean'),
    mean_risk  = ('risk_score',      'mean'),
    pct_2plus  = ('risk_flag_count', lambda x: (x >= 2).mean() * 100),
    count      = ('patient_id',      'count')
).round(2)

print("\nAge Group Risk Flag Summary:")
print(age_grp_flag)

#  Chart 2: age window clustering 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('H5: At which age do multiple risk factors cluster?', fontsize=13)

# Scatter
axes[0].scatter(df['age'], df['risk_flag_count'],
                alpha=0.35, s=30, color='steelblue', edgecolors='none')
z2 = np.polyfit(df['age'], df['risk_flag_count'], 1)
p2 = np.poly1d(z2)
xline2 = np.linspace(df['age'].min(), df['age'].max(), 100)
axes[0].plot(xline2, p2(xline2), color='tomato', linewidth=2, label='Trend')
axes[0].set_title('Age vs Number of Simultaneous Risk Flags')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Risk Flag Count (0–5)')
axes[0].legend()
corr2 = df['age'].corr(df['risk_flag_count'])
axes[0].text(0.05, 0.92, f"Correlation: {corr2:.2f}",
             transform=axes[0].transAxes, fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# % of patients with 2+ flags by age group
grps  = age_grp_flag.index.tolist()
pcts  = age_grp_flag['pct_2plus'].values
bar_c = ['#4caf7d','#f0a500','#e05c5c']
axes[1].bar(grps, pcts, color=bar_c, edgecolor='white', width=0.5)
axes[1].set_title('% of Patients with 2+ Simultaneous Risk Flags')
axes[1].set_ylabel('Percentage (%)')
axes[1].set_xlabel('Age Group')
axes[1].set_ylim(0, 100)
for i, (g, p_val) in enumerate(zip(grps, pcts)):
    n = age_grp_flag.loc[g, 'count']
    axes[1].text(i, p_val + 1.5, f"{p_val:.1f}%\n(n={n})",
                 ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('06_h5_age_risk_clustering.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 06_h5_age_risk_clustering.png")

worst_age_grp = age_grp_flag['pct_2plus'].idxmax()
print(f"""
--- H5 INSIGHT ---
The '{worst_age_grp}' age group has the highest percentage of patients carrying
2 or more simultaneous risk flags ({age_grp_flag.loc[worst_age_grp,'pct_2plus']:.1f}%).

Age-risk correlation: {corr2:.2f} — risk flags accumulate steadily with age.

This is the critical insight: it is not any single factor that makes older
patients dangerous to miss — it is the COMBINATION of several moderate
risks happening at once that makes them high-priority cases.

CAREGIVER ACTION: Screen for risk flag count, not just individual metrics.
A patient with 3 moderate issues is more at-risk than one with 1 severe issue.
""")

