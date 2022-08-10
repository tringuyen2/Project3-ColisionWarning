# Program extracting first column
from re import A
import xlrd
import cv2
 
loc = ("18-6-2022.xlsx")
 
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)


d = dict()

for i in range(sheet.nrows):
    d[int(sheet.cell_value(i, 1))] = int(sheet.cell_value(i, 0))


print(d)

        
video_path = "E:\\Data\\colision-warning\\demo4.mp4"
cap = cv2.VideoCapture(video_path)

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"width: {width}, height: {height}")



out = cv2.VideoWriter("E:\\Data\\colision-warning\\demo4_SA.mp4", cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))


current_time = 85734    # declare time start video
f = 0
times_loop = 0
distance = -1


while True:
    return_value, frame = cap.read()

    f += 1
    if f % fps == 0:
        current_time = str(current_time)
        s0 = current_time[0]
        s1 = current_time[1]
        s2 = current_time[2]
        s3 = current_time[3]
        s4 = current_time[4]

        if current_time[3] == '5' and current_time[4] == '9':
            if current_time[1] == '5' and current_time[2] == '9':
                s0 = str(int(s0) + 1)
                s1 = '0'
                s2 = '0'
                s3 = '0'
                s4 = '0'
            else:
                if s2 == '9':
                    s2 = '0'
                    s1 = str(int(s1) + 1)
                else:
                    s2 = str(int(s2) + 1)
                s3 = '0'
                s4 = '0'
        else:
            if s4 == '9':
                # print("s4 = 9")
                s4 = '0'
                s3 = str(int(s3) + 1)
            else:
                # print("s4 khac 9")
                s4 = str(int(s4) + 1)
        # print(f"s1 = {s1} s2 = {s2} s3 = {s3} s4 = {s4}")
        
        current_time1 = s0 + s1 + s2 +s3 + s4
        current_time = int(current_time1)

        print(current_time)
        if current_time in d:
            distance = d[current_time]
            times_loop = 0
            print(f"distance: {distance}")
        else:
            times_loop += 1

    if times_loop < 7 and distance != -1:
        cv2.putText(frame, str(distance) + " cm", (50, 100), 0, 1.5, (255, 0, 0), 3)


    if return_value:
        out.write(frame)
        cv2.imshow('frame', frame)
    else:
        print('Video has ended or failed, try a different video format!')
        break

    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cv2.destroyAllWindows()











