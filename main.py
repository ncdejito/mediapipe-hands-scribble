# https://google.github.io/mediapipe/solutions/hands.html#python-solution-api
import cv2
import mediapipe as mp
import math
import numpy as np
import vlc

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def smoothen(pts):
    for i in range(3, len(pts)):
        window = [pts[i - 2], pts[i - 1], pts[i]]
        pts[i][0] = np.mean([x for x, _ in window])
        pts[i][1] = np.mean([y for _, y in window])
    return pts


def play_video():
    # https://www.geeksforgeeks.org/vlc-module-in-python-an-introduction/

    media = vlc.MediaPlayer("aspiro.mp4")

    # start playing video
    media.play()
    print("what")
    while True:
        pass


def eudis(coords1, coords2):
    return math.sqrt((coords1[0] - coords2[0]) ** 2 + (coords1[1] - coords2[1]) ** 2)


def is_touching(finger1, finger2, threshold=25):
    return eudis(finger1, finger2) < threshold


scribbles = [[]]
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                INDEX_FINGER_TIP = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                    * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                    * image_height,
                )
                MIDDLE_FINGER_TIP = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
                    * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                    * image_height,
                )
                THUMB_TIP = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
                    * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
                    * image_height,
                )
                PINKY_TIP = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x
                    * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
                    * image_height,
                )

                if is_touching(THUMB_TIP, INDEX_FINGER_TIP):
                    print("Draw mode on")
                    scribbles[-1].append(list(INDEX_FINGER_TIP))
                else:
                    if scribbles[-1] != []:
                        scribbles[-1] = smoothen(scribbles[-1])
                        scribbles.append([])
                if is_touching(THUMB_TIP, MIDDLE_FINGER_TIP):
                    scribbles[-2] = []
                    play_video()
                if is_touching(THUMB_TIP, PINKY_TIP):
                    print("Clear board")
                    scribbles = [[]]
                # draw
                # print(scribbles)
                for scribble in scribbles:
                    pts = np.array(scribble, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    image = cv2.polylines(
                        image, [pts], isClosed=False, color=(255, 191, 0), thickness=5
                    )
                # print(f"Index finger tip coordinates: (", INDEX_FINGER_TIP)
                # mp_drawing.draw_landmarks(
                #     image,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style(),
                # )
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow("MediaPipe Hands", cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
