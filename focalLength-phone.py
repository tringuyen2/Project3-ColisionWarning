import cv2



height = 0.175 #(met)
distance = 0.5 #(met)


img = cv2.imread('focalLength-phone.jpg')
h, w, _ = img.shape
# print(img.shape)
a = (830, 155)
b = (830, 525)

cv2.circle(img, (830, 155), 6, (255, 0, 0), -1)
cv2.circle(img, (830, 525), 6, (255, 0, 0), -1)
cv2.line(img, (830, 155), (830, 525), (255, 255, 0), 1)

print(abs(a[1] - b[1]))
focal_length = ((abs(a[1] - b[1])/h) * distance) / (height)


with open('focalLength-phone.txt', mode='w') as f:
    f.write(str(round(focal_length, 3)))


cv2.imshow('image', img)
cv2.imwrite('focalLength-phone-result.jpg', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

