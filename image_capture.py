
import cv2
import os
import uuid

def capture_images():
    POS_PATH = os.path.join("data", "positive")
    NEG_PATH = os.path.join("data", "negative")
    ANC_PATH = os.path.join("data", "anchor")

    os.makedirs(POS_PATH, exist_ok=True)
    os.makedirs(NEG_PATH, exist_ok=True)
    os.makedirs(ANC_PATH, exist_ok=True)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        frame = frame[120 : 120 + 2250, 200 : 200 + 250, :]
        cv2.imshow("Image collection", frame)

        if cv2.waitKey(1) & 0xFF == ord("a"):
            imgname = os.path.join(ANC_PATH, f"{uuid.uuid1()}.jpg")
            cv2.imwrite(imgname, frame)

        if cv2.waitKey(1) & 0xFF == ord("p"):
            imgname = os.path.join(POS_PATH, f"{uuid.uuid1()}.jpg")
            cv2.imwrite(imgname, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
