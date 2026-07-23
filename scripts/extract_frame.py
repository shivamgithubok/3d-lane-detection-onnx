import cv2
cap = cv2.VideoCapture('videos/input.mp4')
for i in range(60):
    ret, frame = cap.read()
cv2.imwrite('data/images/video_frame_60.jpg', frame)
