import imageio
import os
path = "./imgs/"
num_ite = 8
imgs = os.listdir(path)
images = []
for filename in imgs:
  filename = path + filename
  for i in range(num_ite):
    images.append(imageio.imread(filename))
  
imageio.mimsave('val_score.gif', images)
