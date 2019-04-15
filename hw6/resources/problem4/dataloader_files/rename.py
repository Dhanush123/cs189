import os

directory = '/Users/dhanush/Documents/GitHub/cs189/hw6/resources/problem4/dataloader_files'

for filename in os.listdir(directory):
  with open(filename) as f:
    new_paths = f.read().replace('/content/gdrive/My Drive/CS 189/hw/64code/hw6_mds189/trainval/','/content/gdrive/My Drive/CS 189/hw/64code/hw6_mds189//content/gdrive/My Drive/CS 189/hw/64code/hw6_mds189/trainval/')

  with open(filename, "w") as f:
    f.write(new_paths)
