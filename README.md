<<<<<<< HEAD
# COLISION WARNING IN ADAS base on YOLOv4 and DeepSORT

=======
# COLISION WARNING IN ADAS
(base on YOLOv4 and DeepSORT)
### Video demo: https://www.youtube.com/watch?v=hfMuVcWyqek
>>>>>>> e61a48b9a22484264b47ab05bce2903585d1bbcd


## Install package:
pip install -r requirements.txt

## Run this project:
python object_tracker.py --video ./data/video/test4.mp4 --output ./outputs/test4.avi



### If you want to change focal length for estimating distance, you can edit and run: 
python FocalLength.py --image car-focalLength.png --knownDistanceMet 10 --heightObject 1.6
