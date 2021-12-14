import os
import sys
import glob
import rasterio as rio
import argparse
import numpy as np
from matplotlib import pyplot as plt



def main(args):
    files_to_remove = []
    # handler for keypress events while plot is open
    def on_press(event):
        sys.stdout.flush()
        if event.key == 'y':
            print("Keeping file:", file)
            plt.close(fig)
        if event.key == 'n':
            print("Removing file:", file)
            files_to_remove.append(file)
            plt.close(fig)
        else:
            pass
    
    input_dir = args.input_dir
    files = glob.glob(input_dir + "/**/*.tif", recursive = True)
    for file in files:
        with rio.open(file, driver="GTiff") as src:
            blue = src.read(1)
            green = src.read(2)
            red = src.read(3)

            img = np.dstack((blue, green, red))
            fig, ax = plt.subplots()
            fig.canvas.mpl_connect('key_press_event', on_press)
            ax.set_title(file)
            ax.set_xlabel("Press (Y) to keep, (N) to remove")
            plt.imshow(img)
            plt.show()
    answer = input("Are you sure you want to remove " + str(len(files_to_remove)) + " files? (Y/N): ")
    if answer == 'y' or answer == 'Y':
        for file in files_to_remove:
            os.remove(file)
        print(len(files_to_remove), " files successfully removed.")
    else:
        print("Files will not be removed.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='E:/global_water_dataset',
                        help='path to the directory where the images will be read from')
    args = parser.parse_args()
    main(args)