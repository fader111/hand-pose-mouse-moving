import cv2, time

cam = 2  # 0,1 - какая первая подключилась
cap = cv2.VideoCapture(cam)
while not cap.isOpened(): # await camera 
    pass

while cap.isOpened():
        t0 = time.time()
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow('test frame', frame)
        if cv2.waitKey(10)==27:
              break

cap.release()
cv2.destroyAllWindows()
