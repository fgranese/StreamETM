import os

import pandas as pd
import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from utils.ocpd_utils import online_change_point_detection
import utils.general_utils as gen_ut


def generate_data(subjects_selection, proportions, data, dir_name='documents', time_steps: int = 4,
                  sample_seed: int = 42):
    # Create dictionary from the subjects selection and proportions
    random.seed(sample_seed)
    proportions = dict(zip(subjects_selection, proportions))

    # Initialize time bin and file index
    time_bin = 0
    start_date = datetime(2022, 1, 1)  # Starting date for timestamps
    file_id = 0

    # Initialize a list to collect data for plotting
    plot_data = []

    # Initialize mapping from target to target names
    mapping_targets = {i: name for i, name in enumerate(newsgroups.target_names)}
    rupture_points = []
    used_id_chunk = []

    for j in range(time_steps):
        chunk_size = 3  # np.random.randint(2, 4)  # Generates a number between 2 and 4 inclusive

        for chunk in range(chunk_size):
            combined_chunk = pd.DataFrame(columns=['text', 'target', 'time_bin'])

            used_id_selection = []
            for subject in subjects_selection:
                proportion = proportions[subject][j]
                n = int(proportion * 512)  # Adjust based on proportions for each subject 512
                if n > 0:
                    available_samples = data[data['target'] == subject].shape[0]
                    n = min(n, available_samples)

                    selected_data = data[data['target'] == subject].sample(n, random_state=(file_id + 1) * sample_seed)
                    selected_data = selected_data.copy()  # To avoid SettingWithCopyWarning
                    selected_data['time_bin'] = time_bin
                    combined_chunk = pd.concat([combined_chunk, selected_data], ignore_index=True)
                    used_id_selection += selected_data.index.values[:].tolist()

            if combined_chunk.empty:
                print(f"Chunk {file_id} is empty. Skipping saving.")
                file_id += 1
                time_bin += 1
                continue  # Skip empty chunks

            used_id_chunk += used_id_selection
            if j == 0 and chunk == 0:
                print(used_id_chunk)
            combined_chunk['id'] = range(1, len(combined_chunk) + 1)
            combined_chunk['date'] = [start_date + timedelta(days=k) for k in range(len(combined_chunk))]
            combined_chunk['news'] = combined_chunk['text']

            combined_chunk['keyword'] = " "
            combined_chunk['headline'] = " "

            csv_data = combined_chunk[['id', 'date', 'news', 'keyword', 'headline']]

            # **Modified Filename with Incremented Date**
            # Calculate the timestamp based on start_date and current time_bin
            current_date = start_date + timedelta(days=time_bin)
            # Set hour and minute to '00'
            timestamp = current_date.strftime('%y_%m_%d_00_00')

            os.makedirs(dir_name, exist_ok=True)
            csv_file_name = f"{dir_name}/chunk_{timestamp}.csv"
            os.makedirs(f"{dir_name}/targets", exist_ok=True)
            csv_fine_name_target = f"{dir_name}/targets/chunk_{timestamp}_target.csv"

            csv_data.to_csv(csv_file_name, index=False)
            combined_chunk[['id', 'date', 'news', 'keyword', 'headline', 'time_bin', 'target']].to_csv(
                csv_fine_name_target, index=False
            )
            print(f"Saved: {csv_file_name}")

            plot_data.append(combined_chunk[['time_bin', 'target']])

            file_id += 1
            time_bin += 1

        rupture_points.append(time_bin)

    print(f"Total files saved: {file_id}")
    print(f"Rupture points: {rupture_points[:-1]}, {rupture_points}")

    return plot_data, mapping_targets, time_bin, rupture_points, used_id_chunk


def plot_original_distribution(plot_data, mapping_targets, time_bin, rupture_points: list,
                               plot_file_name='topic_over_time_original.png',
                               change_points_filename: str = 'change_points_chi.npy'):
    # Combine all plot data into a single DataFrame
    final_plot_data = pd.concat(plot_data, ignore_index=True)

    # Group by time_bin and target and count the occurrences
    count_df = final_plot_data.groupby(['time_bin', 'target']).size().reset_index(name='count')

    # Pivot the DataFrame to make it suitable for plotting
    pivot_df = count_df.pivot(index='time_bin', columns='target', values='count').fillna(0)

    # Map the target to the target names
    pivot_df.columns = pivot_df.columns.map(mapping_targets)

    change_points = check_rupture_points_is_correct(rupture_points[:-1], pivot_df.values[:])
    if change_points != rupture_points[:-1]:
        breakpoint()

    batches = list(range(0, time_bin, 3))
    sns.set_theme(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(25, 8))  # Set width and height

    pivot_df.plot(kind='line', figsize=(25, 8))

    # Add vertical lines to show the batches
    for batch in batches:
        plt.axvline(x=batch, color='r', linestyle='--', alpha=0.7)

    for rupture in change_points:
        plt.axvline(x=rupture, color='b', linestyle='-', alpha=0.7)

    # Put the legend outside the plot
    plt.legend(title='Target', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set titles and labels
    plt.title('Target Distribution Over Time Bins', fontsize=16)
    plt.xlabel('Time Bin', fontsize=14)
    plt.ylabel('Count', fontsize=14)

    # Rotate x-axis labels if necessary
    plt.xticks(rotation=45)

    # Enable grid
    plt.grid(True)
    plt.tight_layout()

    np.save(change_points_filename, change_points)

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/' + plot_file_name)
    print(f"Plot saved as {plot_file_name}")
    return pivot_df


def create_proportions(seed: int = 42, n_subjects: int = 10, setting: str = 'easy', time_steps: int = 4):
    assert n_subjects < 20, "Maximum number of subject is 19"

    random.seed(seed)
    subjects_selection_ = random.sample(range(1, 19), n_subjects)
    # n_subject = len(subjects_selection_)

    print(subjects_selection_)

    if setting == 'easy':
        proportions_ = np.full((n_subjects, time_steps), 1 / n_subjects)

    elif setting == 'medium':
        alpha_m = np.random.uniform(2, 10)
        proportions_ = np.random.dirichlet([alpha_m] * n_subjects, size=(time_steps,)).T

    elif setting == 'hard':
        alpha_h = np.random.uniform(0.5, 1)
        proportions_ = np.random.dirichlet([alpha_h] * n_subjects, size=(time_steps,)).T

    elif setting == 'extreme':
        proportions_list = []
        alpha_e = np.random.uniform(2, 10)
        for t in range(time_steps):
            active_topics = np.random.choice([0, 1], size=n_subjects, p=[0.3, 0.7])  # 30% disappear
            raw = np.random.dirichlet(np.ones(n_subjects) * alpha_e) * active_topics
            normalized = raw / raw.sum() if raw.sum() > 0 else raw
            proportions_list.append(normalized)
        proportions_ = np.asarray(proportions_list).T

    elif setting == 'custom':
        # Preselected subjects and proportions
        subjects_selection_ = [7, 14, 19, 13, 9]
        proportions_ = np.array([
            [0.5, 0.1, 0.4, 0.4],
            [0.3, 0.6, 0.0, 0.0],
            [0.1, 0.2, 0.3, 0.1],
            [0.0, 0.0, 0.0, 0.35],
            [0.0, 0.0, 0.2, 0.2]
        ])
    else:
        raise ValueError(setting + ' do not exists')

    print(proportions_)

    return subjects_selection_, proportions_


def find_break_points(threshold, topic_over_time, run_folder, time_step):
    topics_representation = pd.DataFrame({
        'time_bin': [len(topic_over_time) - 1] * len(topic_over_time.columns),
        'topic': list(range(len(topic_over_time.columns))),
        'top_words': [topic.split('.') for topic in list(topic_over_time.columns)]})

    gen_ut.save_to_file(pd.DataFrame(topics_representation), run_folder, f'topics_representation.csv')
    gen_ut.save_to_file(time_step, run_folder, f'time_bin.npy', file_format='npy')
    topic_over_time.columns = range(len(topic_over_time.columns))

    for i in range(len(topic_over_time)):
        topic_over_time_current = topic_over_time.iloc[:i + 1, :]
        gen_ut.save_to_file(topic_over_time_current, run_folder, 'topic_over_time.csv')
        online_change_point_detection(threshold, topic_over_time_current, run_folder, time_step[i])


def check_rupture_points_is_correct(rupture_points_predicted, contingency_table):
    import scipy.stats as stats
    import numpy as np

    # Create the contingency table
    contingency_table = contingency_table.astype('float64')
    contingency_table += 1e-6
    time_bins = len(contingency_table)

    expected_values = np.zeros(contingency_table.shape)
    sum_tb = contingency_table.sum(axis=1)
    sum_c = contingency_table.sum(axis=0)
    total_sum = contingency_table.sum()
    for row in range(len(sum_tb)):
        for col in range(len(sum_c)):
            expected_values[row, col] = sum_tb[row] * sum_c[col] / total_sum

    # Compute Chi-Square test between consecutive time bins
    p_values = []
    for i in range(time_bins - 1):
        contingency_table[i, :].sum()
        chi2, p, _, _ = stats.chi2_contingency([contingency_table[i], contingency_table[i + 1]])
        p_values.append(p)

    # Identify significant change points (where p < 0.05)
    change_points = [i + 1 for i, p in enumerate(p_values) if p < 0.05]

    print("Significant change points:", change_points)
    print("Ruptures points:", rupture_points_predicted)

    return change_points


def generate_single_dataset(n_subject: int, data: pd.DataFrame, time_steps: int, threshold: float,
                            sample_seed: int = 42):

    run_folder = f"data/{setting}"

    subjects_selection, proportions = create_proportions(n_subjects=n_subject, setting=setting, time_steps=time_steps,
                                                         seed=seed)
    data = data.loc[data.target.isin(subjects_selection)]

    plot_data, mapping_targets, time_bin, rupture_points, used_id = generate_data(subjects_selection, proportions, data,
                                                                                  dir_name=run_folder,
                                                                                  sample_seed=sample_seed)
    topic_over_time_original = plot_original_distribution(plot_data, mapping_targets, time_bin, rupture_points,
                                                          plot_file_name=f'topic_over_time_original_{setting}.png',
                                                          change_points_filename=f'{run_folder}/change_points_original.npy')
    find_break_points(threshold, topic_over_time_original, run_folder, np.arange(len(topic_over_time_original)))
    return used_id


if __name__ == '__main__':
    random.seed(384)
    np.random.seed(384)

    setting = 'custom'
    n_subjects = 5
    seed = 197
    time_steps = 4
    threshold = 0.05
    newsgroups = fetch_20newsgroups(subset='all')
    data = pd.DataFrame({'text': newsgroups.data, 'target': newsgroups.target})

    print(setting, n_subjects, seed, time_steps, threshold)
    generate_single_dataset(data=data, n_subject=n_subjects, sample_seed=seed, time_steps=time_steps,
                            threshold=threshold)
