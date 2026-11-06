import pandas as pd
from pathlib import Path

url = 'https://s3.amazonaws.com/openneuro.org/ds004636/suggested_exclusions.csv'
exclusions = pd.read_csv(url)

# Separate the first column into two columns: 'subject' and 'task'
exclusions[['subject', 'task']] = exclusions['Unnamed: 0'].str.split('_', expand=True)
exclusions = exclusions.drop(columns=['Unnamed: 0'])

# Keep only the relevant tasks
relevant_tasks = ['CCTHot', 'discountFix', 'motorSelectiveStop', 'twoByTwo', 'stopSignal']
exclusions = exclusions[exclusions['task'].isin(relevant_tasks)]

# Filter participant
with open('./grace_fmriprep/participants.txt', 'r') as f:
    participants = [line.strip() for line in f]

exclusions = exclusions[exclusions['subject'].isin(participants)]
qc_fail = exclusions[exclusions['MRIQC_fail'] == 1]
qc_fail.to_csv('./grace_fmriprep/qc_fail.csv', index=False)

# Read in fmriprep quality control data
fmri_qc = {}

for file in Path("preprocessed").rglob("*confound*.tsv"):
    print(f"Reading {file.name}...")
    fmri_qc[file.name] = pd.read_csv(file, sep="\t")

# Create a summary of the fMRI QC data
fmri_qc_summary = []

for file_name, df in fmri_qc.items():
    # Extract subject and task from the file name
    subject = file_name.split('_')[0].replace('sub-', '')
    task = file_name.split("_")[2].replace('task-', '')

    # Get the mean framewise displacement (FD) and DVARS
    mean_fd = df['framewise_displacement'].mean()
    max_fd = df['framewise_displacement'].max()
    proportion_high_fd = (df['framewise_displacement'] > 0.5).mean()
    mean_dvars = df['std_dvars'].mean()
    proportion_high_dvars = (df['std_dvars'] > 1.2).mean()
    # Count each frame once if either FD or standardized DVARS exceeds its threshold.
    proportion_high_fd_or_dvars = ((df['framewise_displacement'] > 0.5) |
                                   (df['std_dvars'] > 1.2)).mean()

    # Append to the summary list
    fmri_qc_summary.append({
        'worker_id': subject,
        'task': task,
        'mean_fd': mean_fd,
        'max_fd': max_fd,
        'proportion_high_fd': proportion_high_fd,
        'proportion_high_dvars': proportion_high_dvars,
        'proportion_high_fd_or_dvars': proportion_high_fd_or_dvars,
        'mean_dvars': mean_dvars
    })

# Convert the summary list to a DataFrame
fmri_qc_summary_df = pd.DataFrame(fmri_qc_summary).reset_index(drop=True)

# Define thresholds for QC metrics
threshold_by_prop = {
    'proportion_high_fd_or_dvars': 0.2
}

# Apply the thresholds to determine if a subject fails QC
fmri_qc_summary_df['qc_fail_fmriprep'] = fmri_qc_summary_df['proportion_high_fd_or_dvars'] > threshold_by_prop['proportion_high_fd_or_dvars']
print(f'Number of runs failing motion-based QC: {fmri_qc_summary_df['qc_fail_fmriprep'].sum()}')
print(f'Number of unique subjects failing motion-based QC: {fmri_qc_summary_df[fmri_qc_summary_df['qc_fail_fmriprep']]['worker_id'].nunique()}')

# Combine the QC fail information with the exclusions data
qc_fail = qc_fail.rename(columns={'subject': 'worker_id'})
qc_fail_combined = pd.merge(qc_fail[['worker_id', 'task', 'MRIQC_fail']],
                            fmri_qc_summary_df[['worker_id', 'task', 'qc_fail_fmriprep']], on=['worker_id', 'task'], how='outer')
qc_fail_combined['qc_fail'] = qc_fail_combined['MRIQC_fail'].eq(1) | qc_fail_combined['qc_fail_fmriprep'].eq(True)
# Change to 1/0 representation for qc_fail column
qc_fail_combined['qc_fail'] = qc_fail_combined['qc_fail'].astype(int)
qc_fail_combined = qc_fail_combined.drop(columns=['MRIQC_fail', 'qc_fail_fmriprep'])
qc_fail_combined = qc_fail_combined[qc_fail_combined['qc_fail'] == 1]
qc_fail_combined.to_csv('./Data/qc_fail_combined.csv', index=False)

print(f'Number of runs failing combined QC: {qc_fail_combined.shape[0]}')
print(f'Number of unique subjects failing combined QC: {qc_fail_combined["worker_id"].nunique()}')

