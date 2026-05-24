# Elderly Health Risk Pattern Analysis
## Insight Report

**Dataset:** 300 synthetic elderly patients | 14 features

**Tools:** Python, pandas, Matplotlib, Seaborn

**Goal:** Identify behavioral and clinical patterns that predict
health deterioration in elderly individuals, to support earlier
caregiver intervention.

---

## Key Findings

### H1  Low Activity + Poor Sleep Compound Risk
Patients with *both* low physical activity and poor sleep show
a mean risk score 18.5 points higher than those with neither
factor significantly more than either factor alone. This
suggests these risks multiply, not just add.

**Caregiver action:** Flag patients with both low activity AND
poor sleep for increased monitoring frequency, not just one.

---

### H2  Living Alone Is a Clinical Risk Factor
Elderly patients living alone score 8.3 points higher on risk
than those in assisted care. The gap appears across every
metric : activity, sleep, medication adherence, social
interaction, and fall frequency.

**Caregiver action:** Living condition should be a standard
field in risk screening, not just an administrative detail.

---

### H3  Isolation Risk Compounds With Age
The 85+ age group living alone reaches a mean risk score of
52.4, the highest of any segment in the dataset. Isolation
does not become less dangerous with age; it becomes more so.

**Caregiver action:** Patients aged 85+ living alone are the
single highest-priority monitoring group, even with stable
current vitals.

---

### H4  Cognitive Impairment Drives Non-Adherence
Cognitive impairment patients show a medication adherence rate
of 0.53, 28% below the dataset average of 0.74. Their mean
risk score (52.3) is the highest of any condition group.
Adherence and risk have a correlation of -0.48: non-adherence
is not a symptom, it is a driver.

**Caregiver action:** Cognitive patients need medication-specific
interventions (reminders, simplified regimens), not just
general health monitoring.

---

### H5  Risk Clusters With Age
By age 85+, 63.4% of patients carry 2 or more simultaneous
risk flags compared to 42.9% in the 65�74 group. Age does
not add a single risk factor; it accumulates several moderate
ones simultaneously, creating compound vulnerability that
individual metric screening misses.

**Caregiver action:** Triage should use a composite risk flag
count, not isolated metrics. A patient with 3 moderate issues
is more vulnerable than one with a single severe issue.

---

## Overall Recommendation
A caregiver alert system should prioritise patients who are:
- Aged 85 or above
- Living alone
- Diagnosed with cognitive impairment
- Showing low activity AND poor sleep simultaneously
- Carrying 2 or more simultaneous risk flags

This combination describes the highest-risk cohort in the
dataset and the most likely to deteriorate without early
intervention.

---

*This project is part of a broader elderly care analytics
portfolio. The dataset is synthetically generated to reflect
realistic clinical patterns while avoiding privacy concerns.*
