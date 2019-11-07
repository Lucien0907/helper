import lucienii as ni

files = ni.files_indir(suffix='.nrrd',deep=True,sort_level=-1)

outf = open("train_data_new_flair.cfg", "w")

for line in files:
    outf.write(line)
    outf.write("\n")
outf.close()

