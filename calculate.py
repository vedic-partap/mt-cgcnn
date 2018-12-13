import pandas as pd
import numpy as np

test_results_l = ""
test_results_m = ""
test_results_r = ""

target_saved = "" 

def get_predicted_properties(filename):
    data = pd.read_csv(filename, index_col=False)
    data.iloc(:,2)
    properties = {}
    for x in range(data.shape[1]):
        properties[data.iloc[x,0]] = [float(x) for x in data.iloc[x,2][1:-1].split(',')]
    return properties

def calculate_diff(a,b):
    diff = {}
    for key in a.keys():
        diff[key] = a[key]-b[key]
    return diff

def write_dict(a,filename):
with open('dict.csv', 'w') as csv_file:
    pd.DataFrame.from_dict(data=a).to_csv(filename, header=False)

data_l = get_predicted_properties(test_results_l)
data_m = get_predicted_properties(test_results_m)
data_r = get_predicted_properties(test_results_r)

cal_ml = calculate_diff(data_m,data_l)
cal_mr = calculate_diff(data_m,data_r)
print("Avergare Diff between Original and Reduced values {0}".format(np.sum(cal_ml.values())/len(cal_ml)))
print("Avergare Diff between Original and Increased values {0}".format(np.sum(cal_mr.values())/len(cal_mr)))

write_dict(cal_ml,target_saved+"cal_ml.csv")
write_dict(cal_mr,target_saved+"cal_mr.csv")