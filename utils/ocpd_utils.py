import os

from ocpd.ocpd import install_ocp_package, run_ocpd_analysis, filter_change_points, store_change_points

def online_change_point_detection(threshold, topic_over_time, run_folder, time_step):
    install_ocp_package()
    topic_ids = topic_over_time.columns.tolist()
    topic_over_time = topic_over_time.reset_index(drop=True)
    proportions_csv = os.path.join(run_folder, 'topic_over_time.csv')
    ocpd_csv = os.path.join(run_folder, 'ocpd.csv')
    change_points = run_ocpd_analysis(proportions_csv, threshold)
    change_points = filter_change_points(change_points, topic_over_time)
    store_change_points(change_points, time_step, ocpd_csv, topic_ids)

    return change_points