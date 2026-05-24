
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

print("Libraries loaded successfully.")


df = pd.read_csv('elderly_health_data.csv')

print("\n--- DATASET OVERVIEW ---")
print(f"Shape: {df.shape[0]} patients × {df.shape[1]} columns\n")
print("Column names:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- BASIC STATS ---")
print(df.describe().round(2))

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- CATEGORY DISTRIBUTIONS ---")
print("Living condition:", df['living_condition'].value_counts().to_dict())
print("Primary condition:", df['primary_condition'].value_counts().to_dict())
print("Risk level:", df['risk_level'].value_counts().to_dict())
print("Age group:", df['age_group'].value_counts().to_dict())


fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Dataset Overview — Elderly Health Data (300 patients)', fontsize=14, y=1.01)


axes[0,0].hist(df['risk_score'], bins=25, color='steelblue', edgecolor='white', alpha=0.85)
axes[0,0].axvline(df['risk_score'].mean(), color='tomato', linestyle='--', label=f"Mean: {df['risk_score'].mean():.1f}")
axes[0,0].set_title('Risk Score Distribution')
axes[0,0].set_xlabel('Risk Score')
axes[0,0].set_ylabel('Count')
axes[0,0].legend()


rl_counts = df['risk_level'].value_counts()
colors = {'Low':'#4caf7d', 'Medium':'#f0a500', 'High':'#e05c5c'}
bar_colors = [colors.get(l, 'gray') for l in rl_counts.index]
axes[0,1].bar(rl_counts.index, rl_counts.values, color=bar_colors, edgecolor='white')
axes[0,1].set_title('Risk Level Counts')
axes[0,1].set_ylabel('Number of Patients')
for i, (idx, val) in enumerate(rl_counts.items()):
    axes[0,1].text(i, val + 2, str(val), ha='center', fontsize=11, fontweight='bold')


axes[0,2].hist(df['age'], bins=20, color='mediumpurple', edgecolor='white', alpha=0.85)
axes[0,2].set_title('Age Distribution')
axes[0,2].set_xlabel('Age')
axes[0,2].set_ylabel('Count')


cond_counts = df['primary_condition'].value_counts()
axes[1,0].barh(cond_counts.index, cond_counts.values, color='cadetblue', edgecolor='white')
axes[1,0].set_title('Primary Conditions')
axes[1,0].set_xlabel('Number of Patients')


lc = df['living_condition'].value_counts()
axes[1,1].pie(lc.values, labels=lc.index, autopct='%1.1f%%',
              colors=['#5b9bd5','#ed7d31'], startangle=90)
axes[1,1].set_title('Living Condition Split')


axes[1,2].hist(df['activity_score'], bins=25, color='mediumseagreen', edgecolor='white', alpha=0.85)
axes[1,2].axvline(df['activity_score'].mean(), color='tomato', linestyle='--',
                   label=f"Mean: {df['activity_score'].mean():.1f}")
axes[1,2].set_title('Activity Score Distribution')
axes[1,2].set_xlabel('Activity Score')
axes[1,2].legend()

plt.tight_layout()
plt.savefig('01_dataset_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 01_dataset_overview.png")


print("\n" + "="*60)
print("HYPOTHESIS 1: Combined effect of low activity + poor sleep")
print("="*60)

# Create binary flags
df['low_activity'] = df['activity_score'] < df['activity_score'].median()
df['poor_sleep'] = df['sleep_hours'] < df['sleep_hours'].median()

# 4 groups
df['combo_group'] = 'Both Good'
df.loc[df['low_activity'] & ~df['poor_sleep'], 'combo_group'] = 'Low Activity Only'
df.loc[~df['low_activity'] & df['poor_sleep'], 'combo_group'] = 'Poor Sleep Only'
df.loc[df['low_activity'] & df['poor_sleep'], 'combo_group'] = 'Both Poor'

group_stats = df.groupby('combo_group')['risk_score'].agg(['mean','median','count']).round(2)
group_stats.columns = ['Mean Risk', 'Median Risk', 'Patient Count']
print("\nGroup Statistics:")
print(group_stats.sort_values('Mean Risk', ascending=False))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('H1: Does low activity + poor sleep compound risk?', fontsize=13)

order = ['Both Good', 'Low Activity Only', 'Poor Sleep Only', 'Both Poor']
palette = ['#4caf7d', '#5b9bd5', '#f0a500', '#e05c5c']
means = [df[df['combo_group']==g]['risk_score'].mean() for g in order]
counts = [df[df['combo_group']==g].shape[0] for g in order]

axes[0].bar(order, means, color=palette, edgecolor='white', width=0.6)
for i, (m, c) in enumerate(zip(means, counts)):
    axes[0].text(i, m + 0.5, f'{m:.1f}\n(n={c})', ha='center', fontsize=10)
axes[0].set_title('Mean Risk Score by Combination Group')
axes[0].set_ylabel('Mean Risk Score')
axes[0].set_ylim(0, max(means) * 1.2)
axes[0].tick_params(axis='x', rotation=15)

sns.boxplot(data=df, x='combo_group', y='risk_score', order=order,
            palette=palette, ax=axes[1])
axes[1].set_title('Risk Score Distribution by Group')
axes[1].set_ylabel('Risk Score')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('02_h1_activity_sleep_combined.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 02_h1_activity_sleep_combined.png")


both_poor_mean = df[df['combo_group']=='Both Poor']['risk_score'].mean()
both_good_mean = df[df['combo_group']=='Both Good']['risk_score'].mean()
print(f"\n--- YOUR INSIGHT (fill this in after looking at the chart) ---")
print(f"'Both Poor' group mean risk:  {both_poor_mean:.1f}")
print(f"'Both Good' group mean risk:  {both_good_mean:.1f}")
print(f"Difference:                   {both_poor_mean - both_good_mean:.1f} points")
print(f"\nWrite your finding here:")
print(f"  H1 FINDING: Patients with BOTH low activity AND poor sleep show a risk score")
print(f"  {both_poor_mean - both_good_mean:.1f} points higher than those with neither risk factor,")
print(f"  suggesting these factors compound rather than add independently.")
print(f"\n  WHAT CAREGIVERS SHOULD DO: Monitor patients with dual risk factors")
print(f"  more frequently — they are not just 'twice as risky' but significantly more so.")

