'''
capturing from webcam using openCV
   - https://www.youtube.com/watch?v=CEz6eNvq_jE
drawing hand landmarks using mediapipe
    - https://www.youtube.com/watch?v=RRBXVu5UE-U
accessing specific landmarks to do stuff
    - https://www.youtube.com/watch?v=Ye-lTW68pZc
openCV hand solution docs
    - https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
drawing with openCV
    - https://www.geeksforgeeks.org/python-opencv-cv2-line-method/
'''

import cv2  # computer vision library (get image from webcam)
import mediapipe  # library for interpreting webcam footage (interpret webcam image)
import pyautogui  # control mouse and keyboard
import math


def get_distance(point1, point2):
    return abs(math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2))


# -- OPENCV --
video_capture = cv2.VideoCapture(0)  # 0 indicates 1 camera available (allows access to webcam)
# set dimensions of display window
SCREEN_WIDTH = 2500
SCREEN_HEIGHT = 1600
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
window_title = "hand tracking"

# -- MEDIAPIPE --
mp_hands = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils
hand = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# -- CONTROLS --
touch_threshold = 130  # maximum distance before points are considered touching
action_distance_min = 500  # minimum distance, pinkie and thumb must be apart for actions to register
right_hand = 1  # 1 or -1

click = False
is_clicking = False

right_click = False
is_right_clicking = False

double_click = False
double_click_timer = 0
double_click_max = 6

scroll_increment = 2

move_mouse = False
prev_middle_pos = pyautogui.position()  # position of middle finger tip previous frame/iteration


run = True
while run:
    # get image from camera
    success, frame = video_capture.read()
    if success:

        # analyse camera image for hands
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert colour profile
        result = hand.process(RGB_frame)

        # if found manipulate landmarks
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            # finger coords in terms of frame
            thumb_tip = hand_landmarks.landmark[4]
            thumb_tip = (int(SCREEN_WIDTH*thumb_tip.x), int(SCREEN_HEIGHT*thumb_tip.y))

            index_tip = hand_landmarks.landmark[8]
            index_tip = (int(SCREEN_WIDTH * index_tip.x), int(SCREEN_HEIGHT * index_tip.y))
            index_knuckle = hand_landmarks.landmark[5]
            index_knuckle = (int(SCREEN_WIDTH * index_knuckle.x), int(SCREEN_HEIGHT * index_knuckle.y))

            middle_tip = hand_landmarks.landmark[12]
            middle_tip = (int(SCREEN_WIDTH * middle_tip.x), int(SCREEN_HEIGHT * middle_tip.y))

            ring_tip = hand_landmarks.landmark[16]
            ring_tip = (int(SCREEN_WIDTH * ring_tip.x), int(SCREEN_HEIGHT * ring_tip.y))

            pinkie_tip = hand_landmarks.landmark[20]
            pinkie_tip = (int(SCREEN_WIDTH * pinkie_tip.x), int(SCREEN_HEIGHT * pinkie_tip.y))

            action_distance = get_distance(thumb_tip, pinkie_tip)
            mouse_move_dist = get_distance(thumb_tip, middle_tip)
            scroll_dist = get_distance(index_tip, middle_tip)
            click_dist = get_distance(thumb_tip, index_tip)
            right_click_dist = get_distance(thumb_tip, ring_tip)

            # --- input ---
            # only take action if hand is open and right way around (right hand only)
            if action_distance >= action_distance_min and thumb_tip[0]*right_hand > pinkie_tip[0]*right_hand:

                # move mouse
                if mouse_move_dist < touch_threshold:
                    if not move_mouse:
                        prev_middle_pos = (middle_tip[0], middle_tip[1])  # set as current pos to prevent mouse jumping
                    move_mouse = True
                    mouse = pyautogui.position()
                    new_pos = [0, 0]
                    new_pos[0] = mouse.x + prev_middle_pos[0] - middle_tip[0]  # flip to account for mirroring of direction by webcam
                    new_pos[1] = mouse.y + middle_tip[1] - prev_middle_pos[1]
                    pyautogui.moveTo(new_pos)
                    prev_middle_pos = (middle_tip[0], middle_tip[1])  # update prev pos
                else:
                    move_mouse = False

                # scroll
                if scroll_dist < touch_threshold:
                    if index_tip[1] < thumb_tip[1]:
                        pyautogui.scroll(scroll_increment)
                    if index_tip[1] > thumb_tip[1]:
                        pyautogui.scroll(-scroll_increment)

                # right click
                if right_click_dist < touch_threshold and not is_right_clicking:
                    right_click = True
                    is_right_clicking = True
                elif right_click_dist < touch_threshold and is_right_clicking:
                    right_click = False
                else:
                    is_right_clicking = False
                    right_click = False

                # click
                if click_dist < touch_threshold and not is_clicking:
                    click = True
                    is_clicking = True
                    # if timer not already activated, activate double click timer
                    if double_click_timer == 0:
                        double_click_timer = 1
                    else:
                        double_click = True
                elif click_dist < touch_threshold and is_clicking:
                    click = False
                else:
                    is_clicking = False
                    click = False

                # handle double click
                if double_click_timer > 0:  # increment if timer is activated
                    double_click_timer += 1
                    print(double_click_timer)
                if double_click_timer >= double_click_max:  # if time exceeded, deactivate timer
                    double_click_timer = 0
                if 0 < double_click_timer < double_click_max and double_click:  # if within time and click, double click
                    print("Double click")
                    pyautogui.click(clicks=2, interval=0.2)
                    pyautogui.doubleClick()
                    double_click = False

                # handle right click
                if right_click:
                    pyautogui.click(button="right")

                # handle click
                if click:
                    pyautogui.click()

            # draw hand markers onto image
            for hand_landmark in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
            cv2.line(frame, thumb_tip, index_tip, (255, 0, 0), 3)

        # show image (being the frame from the webcam)
        cv2.imshow(window_title, frame)
        if cv2.waitKey(1) != -1:  # waitKey returns unicode of pressed key
            run = False

cv2.destroyAllWindows()  # ensure all windows are closed
