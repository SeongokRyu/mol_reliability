import csv
import numpy as np

f = open('egfr_chembl_raw.txt')
lines = f.readlines()

with open('egfr_chembl.csv', 'w', newline='') as csvfile:
    fieldnames = ['smiles', 'pIC50', 'activity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for l in lines:
        contents = l.split(',')
        num = len(contents)
        smi = contents[0]
        if num > 2:
            pic50 = np.mean([float(contents[k]) for k in range(1,num-1)])
            label = 1
            if pic50 < 6.0:
                label = 0
            print (smi, pic50, label)
            writer.writerow({'smiles':smi, 'pIC50':pic50, 'activity':label})
