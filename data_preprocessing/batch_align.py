import os
import subprocess

# Paths
sample_data_dir = r"C:\Users\91730\Desktop\Plante_Sentinal_Fusion_Exp\sample_data"
aligned_data_dir = r"C:\Users\91730\Desktop\Plante_Sentinal_Fusion_Exp\aligned_data"
base_img = r"C:\Users\91730\Desktop\Plante_Sentinal_Fusion_Exp\2016_HiRes_Final_Coastline_UTM3N.tif"

# Create the output directory if it doesn't exist
os.makedirs(aligned_data_dir, exist_ok=True)

# Alignment parameters from your base image
target_srs = "EPSG:32603"
pixel_size = 5.532779396951528
te = [598472.146, 7327174.321, 605731.152, 7333144.190] 


for fname in os.listdir(sample_data_dir):
    if fname.lower().endswith('.tif'):
        input_path = os.path.join(sample_data_dir, fname)
        output_path = os.path.join(
            aligned_data_dir, fname.replace('.tif', '_aligned.tif')
    
        )