import pandas as pd
import numpy as np

test_results_l = "../data/result/86/0.99/test_results.csv"
test_results_r = "../data/result/86/1.01/test_results.csv"

target_saved = ""

def get_predicted_properties(filename):
    data = pd.read_csv(filename, index_col=False)
    data.iloc[:,2]
    properties = {}
    for x in range(data.shape[0]):
        properties[data.iloc[x,0]] = [np.array(float(x)) for x in data.iloc[x,2][1:-1].split(',')]
    return properties


def get_predicted_properties_middle(filename):
    data = pd.read_csv(filename, index_col=False)
    data.iloc[:, 2]
    properties = {}
    for x in range(data.shape[0]):
        properties[data.iloc[x, 0]] = [
            np.array(float(x)) for x in data.iloc[x, 1][1:-1].split(',')
        ]
    return properties


def calculate_diff(a,b):
    diff = {}
    sum1,sum2=0,0
    for key in a.keys():
            # print(a[key)
        diff[key] = np.array(a[key][0]-b[key][0],a[key][1]-b[key][1])
        sum1 += a[key][0] - b[key][0]
        sum2 += a[key][1] - b[key][1]
    return diff, sum1/len(diff), sum2/len(diff)

data_l = get_predicted_properties(test_results_l)
data_m1 = get_predicted_properties_middle(test_results_l)
data_m2 = get_predicted_properties_middle(test_results_r)
data_r = get_predicted_properties(test_results_r)

cal_ml,ml_1,ml_2 = calculate_diff(data_m1,data_l)
cal_mr, mr_1, mr_2 = calculate_diff(data_m2,data_r)
print("Avergare Diff between Original and Reduced values {0}, {1}".format(ml_1,ml_2))
print("Avergare Diff between Original and Increased values {0}, {1}".format(mr_1,mr_2))