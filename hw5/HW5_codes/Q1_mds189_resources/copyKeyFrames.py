'''
i used python 2.7 to develop this, i tested it on ubuntu
i also verified it works with python 3.5

you should be able to run this as `python -m copyKeyFrames`

there are a lot of TODOs that are marked that you should address before running
this for your own data. this will put the key frames in a directory you specify.
'''


import json
import os
from glob import glob
import pdb

# TODO: you need to modify the naming format to match your own directory structure
# for me, this copies my key frames to the directory key_frames. obviously the directory
# you specify for the key frames to be copied to needs to exist!
def copy_key_frame(label,f_key,side,calnet_id):
    # NOTE: you may need to change this line to use the python copyfile function, depending on your OS.
    os.system('cp frames/{:s}_{:s}/{:04d}.jpg key_frames/{:s}_{:s}.jpg'.format(label,calnet_id,f_key,label,side))

def read_json(filename):
    with open(filename,'r') as f:
        data = json.load(f)
    return data
    

bilateral_map = {'reach': True, 'squat': True, 'inline': False, 
                 'lunge': False, 'pushup': True, 'hamstrings': False,
                 'stretch': False, 'deadbug': False}

movement_list = ['reach','squat','inline','lunge','pushup','hamstrings','stretch','deadbug']

# TODO: you may wish to change the loadfile naming convention.
loadfile_template = 'labels/{}_{}.json'

# TODO: you'll definitely need to change the calnet_id to match your own, depending on your naming structure.
calnet_id = 'heatherlckwd'

# go through each movement, and copy the relevant frames.
for label in movement_list: #= raw_input('Which movement is this? (this should be one of the bolded labels) ')
    print('Copying relevant frame(s) for: {}'.format(label))
    bilateral = bilateral_map[label]
    loadfile = loadfile_template.format(label,calnet_id)
    
    # load the label data! 
    data = read_json(loadfile)

    # if its a bilateral movement, we only have one key frame, otherwise we have
    # one key frame each for the left and right side motion.
    if bilateral:
        copy_key_frame(label,data['left_frames']['f_key'],'b',calnet_id)
    else:
        copy_key_frame(label,data['left_frames']['f_key'],'l',calnet_id)
        copy_key_frame(label,data['right_frames']['f_key'],'r',calnet_id)
    
    

