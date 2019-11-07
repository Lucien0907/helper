from lucienii import *

files = files_indir(suffix='.nii', include='FLAIR_bias_corrected.nii', path='../data/wmh_data_17-243', deep=True, sort_level=-2)

def e_type(x):
    return np.dtype(x[0][0][0])

for x in files:
    img = nii_to_array(x)
    print(e_type(img))
    resized = resize_slices_cxy(img, (512,512))
    print(e_type(resized))
    resized = np.int16(resized)
    print(e_type(resized))
    save_as_nii(resized, x.rstrip(".nii")+"_resized.nii")
    normalized=normalization(resized)
    print(e_type(normalized))
    normalized = np.int16(normalized)
    print(e_type(normalized))
    save_as_nii(normalized, x.rstrip(".nii")+"_resized_normalized.nii")

print("Finished!")
