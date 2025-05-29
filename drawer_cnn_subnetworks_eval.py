# -*- coding: utf-8 -*-
"""
Created on Thu May 29 10:27:29 2025

@author: 18307
"""

import os
import pandas as pd

# Load the Excel file
path_current = os.getcwd()
# file_path = os.path.join(path_current, 'results_cnn_subnetworks_evaluation', 'cnn_validation_SubRCM_pcc_by_advanced_fm_linear_ratio_rcm.xlsx')
file_path = os.path.join(path_current, 'results_svm_evaluation', 
                         'de_LDS_10_15_evaluation', 
                         'svm_validation_de_LDS_by_advanced_linear_ratio_10_15.xlsx')
excel_file = pd.ExcelFile(file_path)

# Prepare a summary DataFrame
summary_data = []

# Iterate over each sheet and extract the mean metrics from the 17th row
for sheet_name in excel_file.sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    if len(df) >= 17:
        row_data = df.iloc[16].tolist()
        summary_data.append([sheet_name] + row_data)
    else:
        summary_data.append([sheet_name] + ["Insufficient Data"])

# Convert to DataFrame
summary_df = pd.DataFrame(summary_data)

# # Display the summary
# import ace_tools as tools; tools.display_dataframe_to_user(name="Mean Metrics Summary", dataframe=summary_df)
