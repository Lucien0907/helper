from lucienii import *
import time

flairs = files_indir(suffix='nii', path='../data/wmh_data_17-243', include='FLAIR', deep=True)

flairs = change_name(flairs)
        
print(str(len(flairs))+" files found")

for i in range(len(flairs)):
    start=time.time()
    flair = flairs[i]
    corrected = 0
    path = flair.rstrip(flair.split('/')[-1])
    print()
    print("start checking folder No."+str(i)+": \n"+path+":")
    for x in os.listdir(path):
        print(x)
        if x.find("bias_corrected") >= 0:
            print("Already existed, passed")
            corrected = 1
            break
    if corrected == 0:
        correct_bias(flair)
        end=time.time()
        print("Done! Used "+str(end-start)+" Seconds")
