import pandas as pd
from pymatgen import MPRester
API_KEY = "csiNNLAieA61LDbl"
mpr = MPRester(API_KEY)
entries = mpr.query({"elements": {'$in': [u'Ac', u'Al', u'Am', u'Sb', u'Ar', u'As', u'At', u'Au', u'B', u'Ba', u'Be', u'Bh', u'Bi', u'Bk', u'Br', u'C', u'Ca', u'Cd', u'Ce', u'Cf', u'Cl', u'Cm', u'Co', u'Cr', u'Cs', u'Cu', u'Db', u'Dy', u'Er', u'Es', u'Eu', u'F', u'Fe', u'Fm', u'Fr', u'Ga', u'Gd', u'Ge', u'H', u'He', u'Hf', u'Hg', u'Ho', u'Hs', u'I', u'In', u'Ir', u'K', u'Kr', u'La', u'Li', u'Lr', u'Lu', u'Md', u'Mg', u'Mn', u'Mo', u'Mt', u'N', u'Na', u'Nb', u'Nd', u'Ne', u'Ni', u'No', u'Np', u'O', u'Os', u'P', u'Pa', u'Pb', u'Pd', u'Pm', u'Po', u'Pr', u'Pt', u'Pu', u'Ra', u'Rb', u'Re', u'Rf', u'Rh', u'Rn', u'Ru', u'S', u'Ag', u'Sc', u'Se', u'Sg', u'Si', u'Sm', u'Sr', u'Ta', u'Tb', u'Tc', u'Te', u'Th', u'Sn', u'Ti', u'Tl', u'Tm', u'U', u'Uub', u'Uun', u'Uuu', u'V', u'W', u'Xe', u'Y', u'Yb', u'Zn', u'Zr']}}, ["material_id"])
mp_ids = [e['material_id'] for e in entries]
len(set(mp_ids))
data = pd.DataFrame(mp_ids)
data.to_csv("mpids.csv",index=False)
