import pandas as pd
import numpy as np

# file path name notation used l - left - 0.99  r - right - 1.01
test_results_l = "../data/data_86_FE_BG_0.99/results/test_results.csv" # prediction of test result for 99%
mat_hash_l     = "../data/data_86_FE_BG_0.99/material_id_hash.csv"     # material ID hash for 99 %
test_results_r = "../data/data_86_FE_BG_1.01/results/test_results.csv"
mat_hash_r     = "../data/data_86_FE_BG_1.01/material_id_hash.csv"
mp_ids_file    = "mp_ids.csv" # This is list of mpids for which we have to find the difference

# test_results_l ="data/sample/results/1/test_results.csv"  # prediction of test result for 99%
# mat_hash_l =  "data/sample/material_id_hash.csv" # material ID hash for 99 %
# test_results_r = "data/sample/results/1/test_results.csv"
# mat_hash_r = "data/sample/material_id_hash.csv"
# mp_ids_file = "mpids_sample.csv"


"""
This function will creat a map of mpids vs it's predicted properties value
filename : name of file which contains properties
mp_hash_filename : material_id_hash
mp_ids : list of mp_ids
kind = ["predicted","original"] predicted - properties using model, 
								original - properties using dft
"""
def get_properties(filename,mp_hash_filename, mp_ids, kind="predicted"):
	data = pd.read_csv(filename, index_col=False)
	properties = {}
	cif_mp = {}
	mp_hash = pd.read_csv(mp_hash_filename, index_col=False)
	for  i in range(mp_hash.shape[0]):
		cif_mp[mp_hash.iloc[i, 0]] = mp_hash.iloc[i, 1]

	if kind =="predicted":
		for x in range(data.shape[0]):
			if cif_mp[data.iloc[x, 0]] in mp_ids:
				properties[cif_mp[data.iloc[x, 0]]] = [
				np.array(float(x)) for x in data.iloc[x, 2][1:-1].split(',')
				]
	if kind == "original":
		for x in range(data.shape[0]):
			if cif_mp[data.iloc[x, 0]] in mp_ids:
				properties[cif_mp[data.iloc[x, 0]]] = [
				np.array(float(x)) for x in data.iloc[x, 1][1:-1].split(',')
				]
	return properties
"""
calculate difference between original and predicted values
"""
def calculate_diff(a, b):
	diff = {}
	sum1, sum2 = 0, 0
	for key in a.keys():
		# print(a[key)
		diff[key] = np.array(a[key][0] - b[key][0], a[key][1] - b[key][1])
		sum1 += a[key][0] - b[key][0]
		sum2 += a[key][1] - b[key][1]
	return diff, sum1 / len(diff), sum2 / len(diff)


mp_ids = list(pd.read_csv(mp_ids_file, index_col=False).iloc[:,0])
data_l = get_properties(test_results_l, mat_hash_l, mp_ids) #predicted  99%
data_m1 = get_properties(test_results_l, mat_hash_l, mp_ids,"original") # original for 99%
data_m2 = get_properties(test_results_r, mat_hash_r, mp_ids,"original") # original for 101%
data_r = get_properties(test_results_r, mat_hash_r, mp_ids) # predicted for 101%
assert len(data_l) == len(data_m1)
assert len(data_r) == len(data_m2)

cal_ml, ml_1, ml_2 = calculate_diff(data_m1, data_l)
cal_mr, mr_1, mr_2 = calculate_diff(data_m2, data_r)

print(
	"# materials in Reduced Sample is (For which diff will be calculated):{0}"
	.format(len(cal_ml)))
print("Avergare Diff between Original and Reduced values {0}, {1}\n".format(
 ml_1, ml_2))
print(
	"# materials in Increased Sample is (For which diff will be calculated):{0}"
	.format(len(cal_mr)))
print("Avergare Diff between Original and Increased values {0}, {1}".format(
 mr_1, mr_2))
