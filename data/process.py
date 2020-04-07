import csv

prop = 'tpsa'
file_name = 'stock_'+prop
f = open(file_name+'.txt')
lines = f.readlines()

with open(prop+'.csv', 'w', newline='') as csvfile:
    fieldnames = ['smiles', 'prop']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for l in lines:
        smi = l.split(',')[0]
        #prop = l.split(',')[1].strip()
        prop = str(float(l.split(',')[1].strip())/100.0)
        writer.writerow({'smiles':smi, 'prop':prop})

