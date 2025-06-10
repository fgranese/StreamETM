import os
import pandas as pd
from rpy2.robjects import r

from rpy2.robjects.packages import importr

def install_ocp_package():
    try:
        # Attempt to import the package
        ocp = importr('ocp')
    except :
        # If the package is not installed, install it
        utils = importr('utils')
        utils.install_packages('ocp')
        ocp = importr('ocp')

    return ocp

def run_ocpd_analysis(file_path, threshold):
    """Run OCPD analysis using the R 'ocp' library on the given CSV file.

    Args:
        file_path (str): Path to the CSV file containing proportions data.
        threshold (float): The changepoint threshold for the analysis.

    Returns:
        list: A list of change points detected for each column in the CSV.
    """
    R_code = f'''
    library(ocp)
    proportions <- read.csv("{file_path}", sep=",")
    change_points <- list()
    for (i in 1:ncol(proportions)) {{
        data <- as.numeric(proportions[[i]])
        ocpd_res <- onlineCPD(data, missPts = 'mean', cpthreshold = {threshold}, minsep = 2)
        points <- list(ocpd_res$changepoint_lists$threshcps[[1]] - 1)
        change_points <- append(change_points, points)
    }}
    change_points
    '''

    change_points = r(R_code)
    change_points = [list(pts) for pts in change_points]
    change_points = [[int(pt) for pt in pts] for pts in change_points]
    return change_points

def filter_change_points(change_points, pivot_table):
    """Filter out change points that happen when the proportion decreases from t to t+1.

    Args:
        change_points (list): List of change points.
        pivot_table (pd.DataFrame): Dataframe containing proportions data.

    Returns:
        list: Filtered list of change points.
    """
    filtered_points = [
        True if (len(pts) > 0 and pts[-1] == (len(pivot_table) - 1)) else False
        for pts in change_points
    ]
    for i, changepts in enumerate(filtered_points):
        if changepts:
            step = len(pivot_table) - 1
            if pivot_table.iloc[step - 1, i] > pivot_table.iloc[step, i]:
                filtered_points[i] = False
    return filtered_points


def store_change_points(change_points, timestamp, csv_file, topic_ids):
    """Store change points history in a CSV file.

    Args:
        change_points (list): List of change points for each topic.
        timestamp (str): Current timestamp.
        csv_file (str): Path to the CSV file to store history.
        topic_ids (list): List of global topic IDs.
    """
    # Load existing CSV file or create a new DataFrame if it doesn't exist
    if os.path.exists(csv_file):
        change_points_df = pd.read_csv(csv_file)
        # Convert topic columns to integers, ignoring the 'Datetime' column
        change_points_df.columns = ['Datetime'] + [int(col) for col in change_points_df.columns if col != 'Datetime']
    else:
        columns = ['Datetime'] + topic_ids
        change_points_df = pd.DataFrame(columns=columns)

    # Ensure all topic IDs are represented in the DataFrame columns
    existing_topics = set(change_points_df.columns) - {'Datetime'}
    new_topics = set(topic_ids) - existing_topics

    # Add new columns for any newly discovered topics
    for new_topic in new_topics:
        change_points_df[new_topic] = None

    # Create a new row with the current change points
    topic_change_points = dict(zip(topic_ids, change_points))
    new_row = {'Datetime': timestamp}
    new_row.update({tid: topic_change_points.get(tid, None) for tid in topic_ids})

    # Convert the new row dictionary into a DataFrame and concatenate it
    new_row_df = pd.DataFrame([new_row])
    change_points_df = pd.concat([change_points_df, new_row_df], ignore_index=True)

    # Save the updated DataFrame to the CSV file
    change_points_df.to_csv(csv_file, index=False)


def main(proportions_csv, ocpd_csv, pivot_table, timestamp, topic_ids, threshold):
    install_ocp_package()
    change_points = run_ocpd_analysis(proportions_csv, threshold)
    change_points = filter_change_points(change_points, pivot_table)
    store_change_points(change_points, timestamp, ocpd_csv, topic_ids)

