'''
i used python 2.7 to develop this, and i tested it on ubuntu
i also verified it works with python 3.5

you should be able to run this as `python -m createLabelFile` 

there are a lot of TODOs that are marked that you should address before running
this for your own data. this will produce label files in a directory you specify. 
'''


import json
from glob import glob
import pdb
import sys

def get_user_input(query):
    '''
    they changed the user input function from python 2 to 3!
    '''
    if sys.version_info[0] == 2:
        return raw_input(query)
    elif sys.version_info[0] == 3:
        return input(query)

def get_frame_numbers(side,start_frame,end_frame):
    '''
    This is for frame numbers for the movement itself. The start frame of the movement should always be bigger
    than the start frame of the video. The end frame of the movement should always be smaller than the 
    end frame of the video.
    '''
    f_start = int(get_user_input('What is the start frame for the movement with the {} leg in motion? '.format(side)))
    assert f_start > start_frame, 'Oops. Your movement start frame probably is not right at the video start, unless you trimmed it really well!'
    f_end = int(get_user_input('What is the end frame for the movement with the {} leg in motion? '.format(side)))
    assert f_end > f_start, 'Oops. Your end frame needs to be bigger than the start frame.'
    assert end_frame > f_end, 'Oops. You cannot have an end frame for a movement that is at or after the video has ended!'
    f_key = int(get_user_input('What is the key frame for the movement with the {} leg in motion? '.format(side)))
    assert f_key > f_start, 'Oops. Your key frame cannot be before the start frame.'
    assert f_end > f_key, 'Oops. Your key frame cannot be after the end frame.'
    frame_data = {'f_start': f_start, 'f_end': f_end, 'f_key': f_key}
    return frame_data

def write_json(data,savefile):
    with open(savefile,'w') as f:
        json.dump(data,f,indent=4)
    

bilateral_map = {'reach': True, 'squat': True, 'inline': False, 
                 'lunge': False, 'pushup': True, 'hamstrings': False,
                 'stretch': False, 'deadbug': False}

movement_list = ['reach','squat','inline','lunge','pushup','hamstrings','stretch','deadbug']

# TODO: you may wish to change the savefile naming convention.
savefile_template = 'labels/{}_{}.json'

# TODO: you'll definitely need to change the calnet_id to match your own.
calnet_id = 'heatherlckwd'

# go through each movement, and generate the label file.
for label in movement_list: #= get_user_input('Which movement is this? (this should be one of the bolded labels) ')
    print('Generating label file for movement: {}'.format(label))
    bilateral = bilateral_map[label]
    savefile = savefile_template.format(label,calnet_id)
    
    # TODO: you may need to revise the directory this is globbing, depending on your naming convention
    framedir = '{}_{}'.format(label,calnet_id)
    num_frames = len(glob('frames/{}/*.jpg'.format(framedir)))
    assert num_frames > 0, 'Oops. No images found in directory {}!'.format('frames/{}/*.jpg'.format(framedir))
    
    # TODO: update these for your subject. these should be consistent 
    # across all your video label files because you should have the
    # same subject for all your videos, so it's more convenient to change them here. 
    subject_data = {'id':'heatherlckwd', # CalNet ID of subject (if they don't go to Cal, use your own CalNet ID)
                    'height': 167.64,    # barefoot height in cm
                    'torso': 45.,        # from base of neck to pelvis in cm
                    'femur': 47.,        # from hip to knee in cm
                    'wingspan': 131.,    # wrist to wrist wingspan in cm
                    'fitness': 400       # average number of minutes per week working out
                   }

    # input the shortened YouTube link. YOUR YOUTUBE VIDEOS MUST BE UNLISTED OR PUBLIC.
    # this should be of the format https://youtu.be/<youtube_keycode>
    youtube_link = get_user_input('Shortened YouTube link please? ')
    assert youtube_link[:17]=='https://youtu.be/', 'Oops! You did not enter a valid YouTube link!'
    
    # TODO: update these for your recording. only the youtube_link 
    # and num_Frames should be different from one video to another.
    metadata = {'id': calnet_id,        
                'youtube': youtube_link,     # should be of the format https://youtu.be/<youtube_keycode> 
                'height': 1920,              # pixels
                'width': 1080,               # pixels
                'number_frames': num_frames,  
                'fps': 30.,
                'device': 'ios'
               }
    
    start_frame = 1 # remember this is 1-indexed, not 0-indexed!
    end_frame = num_frames
    movement = {'label': label,
                'bilateral': bilateral_map[label],
                'f_start': start_frame,
                'f_end': end_frame
               }

    left_frames = get_frame_numbers('left',start_frame,end_frame)
    if bilateral:
        right_frames = left_frames
    else:
        right_frames = get_frame_numbers('right',start_frame,end_frame)
        assert right_frames['f_key']>left_frames['f_key'], 'Oops! Your right side should have moved after your left side!'
        assert right_frames['f_start']>left_frames['f_end'], 'Oops! Your right side movement should start after your left side movement ends!'
    
    label = {'metadata': metadata,
             'subject_data': subject_data,
             'movement': movement,
             'left_frames': left_frames,
             'right_frames': right_frames
            }
    
    write_json(label,savefile) 
    

