#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from PIL import Image

path = 'E:\\deepika_fyp\\final_year_pro\\data'
path2 = 'E:\\deepika_fyp\\final_year_pro\\data2'  
string1="airplane,apple,banana,bicycle,car,cat,chair,duck,teddy bear,pizza,fire hydrant,train,elephant,knife,cup"
class_names=sorted(list(string1.split(",")))

        

for name in os.listdir(os.path.join(path)):
     for filename in os.listdir(os.path.join(path,name)):
                    im = Image.open(path + '\\' + name + '\\' + filename )
                    new_width  = 100
                    new_height = 100
                    img = im.resize((new_width, new_height), Image.ANTIALIAS)
                    img.save(path2 + '\\'  + name + '\\' + filename ) 


# In[ ]:


import splitfolders

#Split to training, testing and validation folders
splitfolders.ratio('E:\\deepika_fyp\\final_year_pro\\data\\sketch_final', output='E:\\deepika_fyp\\final_year_pro\\data\\sketch_final', seed=1337, ratio=(0.85, 0.1,0.05))

