import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv('/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/data/physionet.org/files/mimiciv/3.1/icu/caregiver.csv')
# df = pd.read_csv('/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/data/physionet.org/files/mimiciv/3.1/hosp/provider.csv')
df = pd.read_csv('/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/data/physionet.org/files/mimiciv/3.1/hosp/admissions.csv')
print(df)
print(df.columns)
print(df['admit_provider_id'].nunique()) # 2045 provider id admit
print(df['subject_id'].nunique()) # 2045 provider id admit

# look on average how many providers per patient
df_a = df.groupby('subject_id').agg({'admit_provider_id': 'nunique', 'admittime': 'count'}).reset_index()
df_a['avg_providers'] = df_a['admit_provider_id'] / df_a['admittime']
df_a = df_a.sort_values(by='avg_providers', ascending=False)
print(df_a[df_a['avg_providers'] < 1])
print(df_a[df_a['avg_providers'] >= 1].sort_values(by='admittime', ascending=False))
print(df_a)
# plot histogram of avg_providers
plt.hist(df_a['avg_providers'], bins=100)
plt.savefig('avg_providers.png')