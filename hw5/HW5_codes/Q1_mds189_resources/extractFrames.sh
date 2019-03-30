# this script was written on ubuntu 16.04, and it was tested on a Mac. i don't currently 
# have a windows machine to test this. when i did use windows.. i used cygwin to mimic 
# the linux terminal for things like this.  

# you need to place this script in the same directory as the directory 
# containing your videos, not in the same directory as your videos.

# to run this script, type ./extractFrames.sh in your terminal and press enter! 
# NOTE: this script will not run without execute permissions. you can do something
# like `chmod +x extractFrames.sh` to appropriately modify permissions.

# TODO: your videos are likely saved in a different directory than mine. replace 'videos/*.mp4'
# with your directory and extension.
for i in videos/*.mp4; do 
   # get the filename. in my case the videos are saved as things like out/squat_panna.mp4
   tmp=${i#*/} # remove the prefix, specifically the out/ 
   fname=${tmp%.*} # remove the suffix, specifically the .mp4
   echo $fname # let's check that we have the right filename, so display it to the terminal 
   # TODO: you'll need to change the root directories below to match where you want to save things
   # specfically, replace /home/panna/Data/cs189/Data/frames with your own directory
   # NOTE: obviously the directory that you put should exist! otherise nothing will get saved!
   mkdir -p /Users/dhanush/Desktop/hw5-exercises/frames/$fname # make the directory where the images will go
   ffmpeg -i "$i" -qscale:v 2 -vsync 0 "/Users/dhanush/Desktop/hw5-exercises/frames/$fname/%04d.jpg" # extract frames
done
