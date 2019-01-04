import sys
from gemmi import cif
import os

# Directory whose cif data has to be changed 
directory = '../data/data_86_FE_BG'
# directory = "data/sample"

# Fraction how much ( e.g. 0.99 mean the final bond length will be ( 1-0.99) * original )
fraction = 0.99
mod_dir = directory+'_'+str(fraction)+'/'

if __name__ == '__main__':
    # make directory if not existed
    if not os.path.exists(mod_dir):
        os.makedirs(mod_dir)
    for file in os.listdir(directory):
        # read all the files and copy if not cif ( like material_id_hash.csv, id_prop.csv etc.)
        if file.endswith(".cif"):
            print(file)
            doc = cif.read_file(directory+"/"+file)
            block = doc.sole_block()
            len_a = str(fraction*float(block.find_pair('_cell_length_a')[1]))
            len_b = str(fraction * float(block.find_pair('_cell_length_b')[1]))
            len_c = str(fraction * float(block.find_pair('_cell_length_c')[1]))
            block.set_pair('_cell_length_a',len_a)
            block.set_pair('_cell_length_b',len_b)
            block.set_pair('_cell_length_c',len_c)
            # saving the modified cif file 
            doc.write_file(mod_dir+file)
        else:
            os.popen('cp ' + directory + "/" + file + ' ' + mod_dir + file)
