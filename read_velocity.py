import cv2
import pytesseract
import re


video_path = "E:\\Data\\colision-warning\\test5.mp4"
file_save = 'save_velocity_test5.txt'


cap = cv2.VideoCapture(video_path)

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"width: {width}, height: {height}")
pytesseract.pytesseract.tesseract_cmd = r'F:\Program files\Tesseract-OCR\tesseract.exe'



f = open(file_save, 'r+')
f.truncate(0)

fr = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        velocity_display = frame[700:718, 340:380]

        
        if (fr % fps == 0 or fr == 0):
            text = pytesseract.image_to_string(velocity_display)
            v = text[:-2]

            v = v.replace("S", "5")
            v = v.replace("i", "1")
            v = v.replace("O", "0")


            v = re.sub("[^0-9]", "", v)


            # if (v[0] == 'S' or v[0] == 's'):
            #     v[0] = '5'
            # if (v[1] == 'S' or v[1] == 's'):
            #     v[1] = '5'   
            # if (v[0] == 'i'):
            #     v[0] = '5'
            with open(file_save, 'a') as f:
                f.write(f'{v}\n')
        cv2.imshow('Frame',velocity_display)

        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        fr += 1
    # Break the loop
    else:
        break

cap.release()
cv2.destroyAllWindows()

