'''
tested with python 2.7 and python 3.5

must be located in the same directory as your compressed file (the one you are submitting!)
check for TODOs, and update appropriately
'''

import json
import os
import pdb
import glob
import numpy as np
import imageio
# NOTE: you may need to install imageio package. if you use pip, you can do so via `pip install imageio` (python 2) or `pip3 install imageio` (python 3)

# TODO: this should be your calnet id, which should be your zip file location, without the .zip extension
filename = 'heatherlckwd'

def check_condition(cond):
    '''
    Convenience function for displaying result of checks.
    '''
    if cond:
        print('Passed!')
    else:
        print('Failed.')

def read_json(filename):
    '''
    We have a lot of jsons to read in this script, so it's convenient
    to include a wrapper function for reading json files.
    '''
    with open(filename,'r') as f:
        data = json.load(f)
    return data

def check_kps_file(exportfile,filename):
    # all the possible labels for the keypoint annotations
    label_keys_full = ['REye', 'REar','LKnee', 'LWrist', 'LSmallToe', 'RShoulder', 
                       'LAnkle', 'RKnee', 'LEar', 'RBigToe', 'RAnkle', 'LElbow', 
                       'RElbow', 'LBigToe', 'RHeel', 'RWrist', 'RSmallToe', 'RHip', 
                       'Nose', 'MidHip', 'which_leg_is_in_motion?', 'LShoulder', 
                       'LEye', 'LHip', 'Neck', 'LHeel', 'what_movement_is_this?']
    # get the data in the labelbox export file
    data = read_json(exportfile)
    # verify you labeled 13 images
    num_labels = len(data) == 13
    # checks that image you labeled is in your key_frames directory
    image_exists = True 
    # checks that all the keypoint annotations are labeled according to the specified format
    label_format = True
    # in check_label_file, we will verify that the image size matches what you put in your label file
    imshape_dict = {'squat':0,'reach':0,'pushup':0,'inline':0,
                    'hamstrings':0,'stretch':0,'lunge':0,'deadbug':0}
    imlist = []
    # loop through each labeled key frame
    for da in data:
        # check that External ID image lives in key_frames
        image_exists = image_exists and os.path.isfile('{}/key_frames/{}'.format(filename,str(da['External ID'])))
        assert image_exists, 'Oops! You are missing a file named {} from your key_frames directory'.format(str(da['External ID']))
        # read the image
        im = np.asarray(imageio.imread('{}/key_frames/{}'.format(filename,str(da['External ID']))))
        imshape_dict[da['Label']['what_movement_is_this?']] = im.shape
        imlist.append(tuple(np.ndarray.flatten(im).tolist()))
        label_keys_i = [str(k) for k in da['Label'].keys()]
        # check that the movements and leg motions are labeled
        movement = 'what_movement_is_this?' in label_keys_i and 'which_leg_is_in_motion?' in label_keys_i
        assert movement, 'Oops! You should have labeled the movement and the leg.'
        # check that the other label keys make sense
        for k in label_keys_i:
            label_format = label_format and k in label_keys_full 
            assert label_format, 'Oops! How did {} end up as a label in your labelbox export?'.format(k)
    # check all the images are different
    imlist_cond = len(set(imlist)) == 13
    assert imlist_cond, 'Oops! You should have exactly 13 unique key frames.'

    return num_labels and image_exists and label_format and imlist_cond, imshape_dict
    
def check_label_file(labfile,filename):
    print('Checking label file {}'.format(labfile))
    data = read_json(labfile) 
    # check that youtube entry is a link
    youtube_cond = data['metadata']['youtube'][:17]=='https://youtu.be/'
    assert youtube_cond, 'Oops! Your YouTube link should be a shortened YouTube link that begins like https://youtu.be/'
    if data['movement']['bilateral']:
        assert data['movement']['label'] in ['squat','reach','pushup'], 'Oops! Only squat, reach, and pushup are the bilateral movements'
        left_before_right_cond = True
        bilateral_cond = data['left_frames'] == data['right_frames']
        assert bilateral_cond, 'Oops! Your left_frames and right_frames in the label file should be idenical for bilateral movements'
    else:
        assert data['movement']['label'] in ['inline','hamstrings','lunge','deadbug','stretch'], 'Oops! Only inline, hamstrings, lunge, deadbug, and stretch are the unilateral movements'
        bilateral_cond = data['left_frames'] != data['right_frames']
        assert bilateral_cond, 'Oops! Your left_frames and right_frames in the label file must be different for unilateral movements'
        left_before_right_cond = (data['left_frames']['f_key'] < data['right_frames']['f_key']) and (data['left_frames']['f_end'] < data['right_frames']['f_start'])
        assert left_before_right_cond, 'Oops! You should have recorded your left leg motion before your right leg motion for unilateral movements'
    left_fkey_limits_cond = (data['left_frames']['f_start'] < data['left_frames']['f_key']) and (data['left_frames']['f_key'] < data['left_frames']['f_end'])
    assert left_fkey_limits_cond, 'Oops! Your left_frames f_key label needs to be between your left_frames f_start and f_end labels'
    right_fkey_limits_cond = (data['right_frames']['f_start'] < data['right_frames']['f_key']) and (data['right_frames']['f_key'] < data['right_frames']['f_end'])
    assert right_fkey_limits_cond, 'Oops! Your right_frames f_key label needs to be between your right_frames f_start and f_end labels'
    left_limits_cond = (data['left_frames']['f_start'] >= data['movement']['f_start']) and (data['left_frames']['f_end'] <= data['movement']['f_end'])
    assert left_limits_cond, 'Oops! Your left_frames cannot start before your video, and cannot end after your video'
    right_limits_cond = (data['right_frames']['f_start'] >= data['movement']['f_start']) and (data['right_frames']['f_end'] <= data['movement']['f_end'])
    assert right_limits_cond, 'Oops! Your right_frames cannot start before your video, and cannot end after your video'
    start_before_end_cond = (data['left_frames']['f_start'] < data['left_frames']['f_end']) and (data['right_frames']['f_start'] < data['right_frames']['f_end']) and (data['movement']['f_start']<data['movement']['f_end']) 
    assert start_before_end_cond, 'Oops! Somewhere you have a labeled start frame that does not come before a labeled end frame'
    no_zero_start_cond = data['movement']['f_start'] > 0
    assert no_zero_start_cond, 'Oops! Remember the frame labels need to be 1-indexed meaning the first frame has index 1, not 0'
    one_start_cond = data['movement']['f_start'] == 1
    assert one_start_cond, 'Oops! data[movement][f_start] needs to be 1'
    movement_end_cond = data['movement']['f_end'] == data['metadata']['number_frames']
    assert movement_end_cond, 'Oops! data[movement][f_end] needs to match the number of frames'
    # check that label is in label list
    label_valid_cond = data['movement']['label'] in ['deadbug','pushup','lunge','inline','squat','reach','stretch','hamstrings']
    assert label_valid_cond, 'Oops! Your video label must be one of the eight specified labels'
    # check that fps = 30
    fps_cond = int(data['metadata']['fps']) == 30
    assert fps_cond, 'Oops! Your videos must be at 30fps. If you did not record at 30fps, you need to re-code your videos to 30fps. This can be done easily with ffmpeg'
    label_cond = youtube_cond and bilateral_cond and left_before_right_cond and start_before_end_cond and no_zero_start_cond and left_limits_cond and right_limits_cond and label_valid_cond and fps_cond and left_fkey_limits_cond and right_fkey_limits_cond and movement_end_cond and one_start_cond
    # check that you didn't just use the default subject data..
    default_subject_data = subject_data = {'id':'heatherlckwd', 
                    'height': 167.64,    # barefoot height in cm
                    'torso': 45.,        # from base of neck to pelvis in cm
                    'femur': 47.,        # from hip to knee in cm
                    'wingspan': 131.,    # wrist to wrist wingspan in cm
                    'fitness': 400       # average number of minutes per week working out
                   }
    subject_cond = data['subject_data'] != default_subject_data 
    assert subject_cond, 'Oops! Did you update your subject data in your label file to reflect the subject in your video??'
    # check that your metadata id and zip file filename match. these should both be your calnet id
    id_cond = data['metadata']['id'] == filename 
    assert id_cond, 'Oops! Your metadata id and zip filename should match, and they should both be your own calnet id'
    label_cond = label_cond and subject_cond and id_cond
    
    return label_cond,data['metadata']['youtube'],data['movement']['label'],data['metadata']['height'],data['metadata']['width']

print('Running data check script on {}.zip'.format(filename))
assert filename != 'heatherlckwd', 'Oops! You need to change the filename variable in line 18 to match your zip file'

print('Checking that the compressed file has .zip extension......')
cond = os.path.isfile('{}.zip'.format(filename))
assert cond, 'Oops! Where is your zip file?'
check_condition(cond)

print('Unzipping {}.zip'.format(filename))
os.system('unzip {}.zip'.format(filename))

print('Checking directory structure of unpacked folder......')
labels_exists = os.path.isdir('{}/labels'.format(filename))
assert labels_exists, 'Oops! Where is your labels folder?'
key_frames_exists = os.path.isdir('{}/key_frames'.format(filename))
assert key_frames_exists, 'Oops! Where is your key_frames folder?'
subfolders = [f for f in os.listdir(filename) if not f.startswith('.')]
cond = labels_exists and key_frames_exists and len(subfolders)==2
assert cond, 'Oops! You should only have two subfolders in your submission: labels and key_frames'
check_condition(cond)

print('Checking key frames......')
imlist = glob.glob('{}/key_frames/*.jpg'.format(filename))
cond = (len(imlist)==13)
assert cond, 'Oops! You should have 13 key frames'
check_condition(cond)

print('Checking keypoints label file......')
lablist = glob.glob('{}/labels/*.json'.format(filename))
assert len(lablist)==9, 'Oops! You should have 9 label files'
# should have one keypoints export file
kps_export = glob.glob('{}/labels/export*.json'.format(filename))
assert len(kps_export) == 1, 'Oops. You should have exactly 1 label file from labelbox'
# read kps file, and check something..
cond_kps,imshape_dict = check_kps_file(kps_export[0],filename)
check_condition(cond_kps)

print('Checking video label files......')
imshape_dict_compare = {'squat':0,'reach':0,'pushup':0,'inline':0,
                        'hamstrings':0,'stretch':0,'lunge':0,'deadbug':0}
label_cond = True
youtube_list = []
label_list = []
for labfile in lablist:
    # as long as we aren't looking at the labelbox file.. 
    if labfile != kps_export[0]:
        # verify individual label file
        labcond_i, youtube_link, label_i, height_i, width_i = check_label_file(labfile,filename)
        imshape_dict_compare[label_i] = (height_i,width_i,3)
        label_cond = label_cond and labcond_i
        assert label_cond, 'Oops! Something went wrong with {}'.format(labfile)
        youtube_list.append(youtube_link)
        label_list.append(label_i)

# check whether the image sizes in your label file match the actual image sizes
assert imshape_dict == imshape_dict_compare, 'Oops! Did you forget to update your label files with the correct image size?'
cond_imsize_match = imshape_dict == imshape_dict_compare

assert len(label_list)==8 and len(set(label_list))==8, 'Oops! You should have 8 different movements labeled'
assert len(youtube_list)==8 and len(set(youtube_list))==8, 'Oops! You should have 8 different YouTube links'

cond_label_files = (len(set(label_list)) == 8) and (len(set(youtube_list)) == 8) and label_cond and (len(label_list)==8) and (len(youtube_list)==8) and (imshape_dict == imshape_dict_compare)
check_condition(cond_label_files)

