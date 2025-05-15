'''
import cv2
import mediapipe as mp
import pyautogui
import time

# Constants for drawing
CIRCLE_RADIUS = 5
CIRCLE_COLOR = (0, 255, 0)  # Green
CIRCLE_THICKNESS = -1  # Filled circle
LINE_COLOR = (0, 255, 0)  # Green
LINE_THICKNESS = 2
SCALING_FACTOR = 5.0  # Factor to amplify the cursor movement


def init_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video device.")
    return cap


def process_frame(frame):
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, rgb_frame


def draw_landmarks(frame, hands, drawing_utils):
    for hand in hands:
        drawing_utils.draw_landmarks(frame, hand)
        landmarks = hand.landmark
        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)
        for connection in mp.solutions.hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_x = int(landmarks[start_idx].x * frame.shape[1])
            start_y = int(landmarks[start_idx].y * frame.shape[0])
            end_x = int(landmarks[end_idx].x * frame.shape[1])
            end_y = int(landmarks[end_idx].y * frame.shape[0])
            cv2.line(frame, (start_x, start_y), (end_x, end_y), LINE_COLOR, LINE_THICKNESS)
    return landmarks


def get_landmark_coordinates(landmarks, frame_width, frame_height):
    coords = {}
    for id, landmark in enumerate(landmarks):
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        coords[id] = (x, y)
    return coords


def map_to_screen(coords, screen_width, screen_height, frame_width, frame_height):
    mapped_coords = {}
    for id, (x, y) in coords.items():
        mapped_x = screen_width * x / frame_width
        mapped_y = screen_height * y / frame_height
        mapped_coords[id] = (mapped_x, mapped_y)
    return mapped_coords


def move_cursor(index_coords, plocx, plocy, smoothening):
    index_x, index_y = index_coords
    clocx = plocx + (index_x - plocx) / smoothening
    clocy = plocy + (index_y - plocy) / smoothening

    # Apply scaling factor for more sensitive movement
    clocx = plocx + (clocx - plocx) * SCALING_FACTOR
    clocy = plocy + (clocy - plocy) * SCALING_FACTOR

    pyautogui.moveTo(clocx, clocy)
    return clocx, clocy


def detect_gestures(coords, thumb_coords, click_time, click_threshold, single_click_flag, left_dragging):
    thumb_x, thumb_y = thumb_coords

    # Left click
    if abs(coords[8][1] - thumb_y) < 70:
        current_time = time.time()
        if current_time - click_time < click_threshold:
            pyautogui.doubleClick()
            click_time = 0
        else:
            if not single_click_flag:
                pyautogui.click()
                single_click_flag = True
            click_time = current_time
    else:
        single_click_flag = False

    # Drag
    if abs(coords[16][1] - thumb_y) < 70:
        if not left_dragging:
            pyautogui.mouseDown()
            left_dragging = True
    else:
        if left_dragging:
            pyautogui.mouseUp()
            left_dragging = False

    # Right click
    if abs(coords[12][1] - thumb_y) < 70:
        pyautogui.rightClick()
        pyautogui.sleep(1)

    # Scroll
    extended_fingers = [id for id in [8, 12, 16, 20] if coords[id][1] < coords[id - 2][1]]

    if len(extended_fingers) == 0:  # Only thumb is extended
        thumb_tip_y = coords[4][1]
        wrist_y = coords[0][1]

        if thumb_tip_y < wrist_y - 40:  # Thumbs up gesture
            pyautogui.scroll(200)  # Increase this value to scroll faster
        elif thumb_tip_y > wrist_y + 40:  # Thumbs down gesture
            pyautogui.scroll(-200)  # Increase this value to scroll faster

    return click_time, single_click_flag, left_dragging


def add_user_instructions(frame):
    instructions = [
        "Virtual Mouse Instructions:",
        "1. Move cursor: Use Index Finger",
        "2. Left Click: Bring Thumb close to Index Finger",
        "3. Right Click: Bring Thumb close to Middle Finger",
        "4. Drag: Hold Thumb close to Ring Finger",
        "5. Scroll: Thumbs Up to Scroll Up, Thumbs Down to Scroll Down"
    ]
    y0, dy = 20, 30
    for i, line in enumerate(instructions):
        y = y0 + i * dy
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def main():
    cap = init_webcam()
    hand_detector = mp.solutions.hands.Hands()
    drawing_utils = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()
    smoothening = 7
    plocx, plocy = 0, 0
    click_time = 0
    click_threshold = 0.3
    single_click_flag = False
    left_dragging = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame, rgb_frame = process_frame(frame)
        frame_height, frame_width, _ = frame.shape
        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        if hands:
            landmarks = draw_landmarks(frame, hands, drawing_utils)
            coords = get_landmark_coordinates(landmarks, frame_width, frame_height)
            mapped_coords = map_to_screen(coords, screen_width, screen_height, frame_width, frame_height)
            clocx, clocy = move_cursor(mapped_coords[8], plocx, plocy, smoothening)
            plocx, plocy = clocx, clocy

            click_time, single_click_flag, left_dragging = detect_gestures(
                mapped_coords, mapped_coords[4], click_time, click_threshold, single_click_flag, left_dragging
            )

        add_user_instructions(frame)
        cv2.imshow('Virtual Mouse', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
'''

'''
import cv2
import mediapipe as mp
import pyautogui
import time

# Constants for drawing
CIRCLE_RADIUS = 5
CIRCLE_COLOR = (0, 255, 255)  # Neon Sky Blue
CIRCLE_THICKNESS = -1  # Filled circle
LINE_COLOR = (0, 255, 255)  # Neon Sky Blue
LINE_THICKNESS = 2
SCALING_FACTOR = 5.0  # Factor to amplify the cursor movement


def init_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video device.")
    return cap


def process_frame(frame):
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, rgb_frame


def draw_landmarks(frame, hands, drawing_utils):
    for hand in hands:
        drawing_utils.draw_landmarks(frame, hand)
        landmarks = hand.landmark
        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)
        for connection in mp.solutions.hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_x = int(landmarks[start_idx].x * frame.shape[1])
            start_y = int(landmarks[start_idx].y * frame.shape[0])
            end_x = int(landmarks[end_idx].x * frame.shape[1])
            end_y = int(landmarks[end_idx].y * frame.shape[0])
            cv2.line(frame, (start_x, start_y), (end_x, end_y), LINE_COLOR, LINE_THICKNESS)
    return landmarks


def get_landmark_coordinates(landmarks, frame_width, frame_height):
    coords = {}
    for id, landmark in enumerate(landmarks):
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        coords[id] = (x, y)
    return coords


def map_to_screen(coords, screen_width, screen_height, frame_width, frame_height):
    mapped_coords = {}
    for id, (x, y) in coords.items():
        mapped_x = screen_width * x / frame_width
        mapped_y = screen_height * y / frame_height
        mapped_coords[id] = (mapped_x, mapped_y)uj
    return mapped_coords


def move_cursor(index_coords, plocx, plocy, smoothening):
    index_x, index_y = index_coords
    clocx = plocx + (index_x - plocx) / smoothening
    clocy = plocy + (index_y - plocy) / smoothening

    # Apply scaling factor for more sensitive movement
    clocx = plocx + (clocx - plocx) * SCALING_FACTOR
    clocy = plocy + (clocy - plocy) * SCALING_FACTOR

    pyautogui.moveTo(clocx, clocy)
    return clocx, clocy


def detect_gestures(coords, thumb_coords, click_time, click_threshold, single_click_flag, left_dragging):
    thumb_x, thumb_y = thumb_coords

    # Left click
    if abs(coords[8][1] - thumb_y) < 70:
        current_time = time.time()
        if current_time - click_time < click_threshold:
            pyautogui.doubleClick()
            click_time = 0
        else:
            if not single_click_flag:
                pyautogui.click()
                single_click_flag = True
            click_time = current_time
    else:
        single_click_flag = False

    # Drag
    if abs(coords[16][1] - thumb_y) < 70:
        if not left_dragging:
            pyautogui.mouseDown()
            left_dragging = True
    else:
        if left_dragging:
            pyautogui.mouseUp()
            left_dragging = False

    # Right click
    if abs(coords[12][1] - thumb_y) < 70:
        pyautogui.rightClick()
        time.sleep(1)

    # Scroll
    extended_fingers = [id for id in [8, 12, 16, 20] if coords[id][1] < coords[id - 2][1]]

    if len(extended_fingers) == 0:  # Only thumb is extended
        thumb_tip_y = coords[4][1]
        wrist_y = coords[0][1]

        if thumb_tip_y < wrist_y - 40:  # Thumbs up gesture
            pyautogui.scroll(200)  # Increase this value to scroll faster
        elif thumb_tip_y > wrist_y + 40:  # Thumbs down gesture
            pyautogui.scroll(-200)  # Increase this value to scroll faster

    return click_time, single_click_flag, left_dragging


def add_user_instructions(frame):
    instructions = [
        "Virtual Mouse Instructions:",
        "1. Move cursor: Use Index Finger",
        "2. Left Click: Bring Thumb close to Index Finger",
        "3. Right Click: Bring Thumb close to Middle Finger",
        "4. Drag: Hold Thumb close to Ring Finger",
        "5. Scroll: Thumbs Up to Scroll Up, Thumbs Down to Scroll Down"
    ]
    y0, dy = 20, 30
    for i, line in enumerate(instructions):
        y = y0 + i * dy
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


def main():
    cap = init_webcam()
    hand_detector = mp.solutions.hands.Hands()
    drawing_utils = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()
    smoothening = 7
    plocx, plocy = 0, 0
    click_time = 0
    click_threshold = 0.3
    single_click_flag = False
    left_dragging = False

    # Allow window resizing
    cv2.namedWindow('Virtual Mouse', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame, rgb_frame = process_frame(frame)
        frame_height, frame_width, _ = frame.shape
        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        if hands:
            hand = hands[0]
            landmarks = draw_landmarks(frame, [hand], drawing_utils)
            coords = get_landmark_coordinates(landmarks, frame_width, frame_height)
            mapped_coords = map_to_screen(coords, screen_width, screen_height, frame_width, frame_height)
            clocx, clocy = move_cursor(mapped_coords[8], plocx, plocy, smoothening)
            plocx, plocy = clocx, clocy

            click_time, single_click_flag, left_dragging = detect_gestures(
                mapped_coords, mapped_coords[4], click_time, click_threshold, single_click_flag, left_dragging
            )

        add_user_instructions(frame)

        # Dynamically resize frame to current window size
        new_width = cv2.getWindowImageRect('Virtual Mouse')[2]
        new_height = cv2.getWindowImageRect('Virtual Mouse')[3]
        frame = cv2.resize(frame, (new_width, new_height))

        cv2.imshow('Virtual Mouse', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

'''




'''
import cv2
import mediapipe as mp
import pyautogui
import time

# Constants for drawing
CIRCLE_RADIUS = 5
CIRCLE_COLOR = (255, 255, 0)  # Neon Sky Blue
CIRCLE_THICKNESS = -1
LINE_COLOR = (255, 255, 0)  # Neon Sky Blue
LINE_THICKNESS = 2
SCALING_FACTOR = 5.0

def init_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video device.")
    return cap

def process_frame(frame):
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, rgb_frame

def draw_landmarks(frame, hands, drawing_utils):
    for hand in hands:
        drawing_utils.draw_landmarks(frame, hand)
        landmarks = hand.landmark
        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)
        for connection in mp.solutions.hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_x = int(landmarks[start_idx].x * frame.shape[1])
            start_y = int(landmarks[start_idx].y * frame.shape[0])
            end_x = int(landmarks[end_idx].x * frame.shape[1])
            end_y = int(landmarks[end_idx].y * frame.shape[0])
            cv2.line(frame, (start_x, start_y), (end_x, end_y), LINE_COLOR, LINE_THICKNESS)
    return landmarks

def get_landmark_coordinates(landmarks, frame_width, frame_height):
    coords = {}
    for id, landmark in enumerate(landmarks):
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        coords[id] = (x, y)
    return coords

def map_to_screen(coords, screen_width, screen_height, frame_width, frame_height):
    mapped_coords = {}
    for id, (x, y) in coords.items():
        mapped_x = screen_width * x / frame_width
        mapped_y = screen_height * y / frame_height
        mapped_coords[id] = (mapped_x, mapped_y)
    return mapped_coords

def move_cursor(index_coords, plocx, plocy, smoothening):
    index_x, index_y = index_coords
    clocx = plocx + (index_x - plocx) / smoothening
    clocy = plocy + (index_y - plocy) / smoothening

    clocx = plocx + (clocx - plocx) * SCALING_FACTOR
    clocy = plocy + (clocy - plocy) * SCALING_FACTOR

    pyautogui.moveTo(clocx, clocy)
    return clocx, clocy

def detect_gestures(coords, thumb_coords, click_time, click_threshold, single_click_flag, left_dragging):
    thumb_x, thumb_y = thumb_coords

    if abs(coords[8][1] - thumb_y) < 70:
        current_time = time.time()
        if current_time - click_time < click_threshold:
            pyautogui.doubleClick()
            click_time = 0
        else:
            if not single_click_flag:
                pyautogui.click()
                single_click_flag = True
            click_time = current_time
    else:
        single_click_flag = False

    if abs(coords[16][1] - thumb_y) < 70:
        if not left_dragging:
            pyautogui.mouseDown()
            left_dragging = True
    else:
        if left_dragging:
            pyautogui.mouseUp()
            left_dragging = False

    if abs(coords[12][1] - thumb_y) < 70:
        pyautogui.rightClick()
        time.sleep(1)

    extended_fingers = [id for id in [8, 12, 16, 20] if coords[id][1] < coords[id - 2][1]]

    if len(extended_fingers) == 0:
        thumb_tip_y = coords[4][1]
        wrist_y = coords[0][1]
        if thumb_tip_y < wrist_y - 40:
            pyautogui.scroll(200)
        elif thumb_tip_y > wrist_y + 40:
            pyautogui.scroll(-200)

    return click_time, single_click_flag, left_dragging

def add_user_instructions(frame):
    overlay = frame.copy()
    alpha = 0.6
    cv2.rectangle(overlay, (5, 5), (500, 200), (0, 0, 0), -1)  # Semi-transparent black
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    instructions = [
        "Virtual Mouse Instructions:",
        "1. Move cursor: Use Index Finger",
        "2. Left Click: Thumb close to Index",
        "3. Right Click: Thumb close to Middle",
        "4. Drag: Thumb close to Ring Finger",
        "5. Scroll: Thumbs Up/Down"
    ]
    y0, dy = 30, 30
    for i, line in enumerate(instructions):
        y = y0 + i * dy
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    return frame

def main():
    cap = init_webcam()
    hand_detector = mp.solutions.hands.Hands()
    drawing_utils = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()
    smoothening = 7
    plocx, plocy = 0, 0
    click_time = 0
    click_threshold = 0.3
    single_click_flag = False
    left_dragging = False

    cv2.namedWindow('Virtual Mouse', cv2.WINDOW_NORMAL)

    DPI_SCALE = 10  # 1000 DPI / 100 (normal) = 10x bigger

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame, rgb_frame = process_frame(frame)
        frame_height, frame_width, _ = frame.shape
        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        if hands:
            hand = hands[0]
            landmarks = draw_landmarks(frame, [hand], drawing_utils)
            coords = get_landmark_coordinates(landmarks, frame_width, frame_height)
            mapped_coords = map_to_screen(coords, screen_width, screen_height, frame_width, frame_height)
            clocx, clocy = move_cursor(mapped_coords[8], plocx, plocy, smoothening)
            plocx, plocy = clocx, clocy

            click_time, single_click_flag, left_dragging = detect_gestures(
                mapped_coords, mapped_coords[4], click_time, click_threshold, single_click_flag, left_dragging
            )

        frame = add_user_instructions(frame)

        # Upscale frame for high "DPI" look
        frame = cv2.resize(frame, (frame.shape[1]*DPI_SCALE, frame.shape[0]*DPI_SCALE))

        cv2.imshow('Virtual Mouse', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
'''

'''
import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import random

# Constants for drawing
CIRCLE_RADIUS = 5
CIRCLE_COLOR = (255, 255, 0)  # Neon Sky Blue
CIRCLE_THICKNESS = -1  # Filled circle
LINE_COLOR = (255, 255, 0)
LINE_THICKNESS = 2
SCALING_FACTOR = 5.0

# Mesh settings
NUM_POINTS = 25  # Light mesh
POINT_SPEED = 0.5
CONNECT_DIST = 150

# Initialize points for moving mesh
points = []
for _ in range(NUM_POINTS):
    points.append({
        "pos": np.array([random.uniform(0, 1), random.uniform(0, 1)]),
        "vel": np.random.uniform(-POINT_SPEED, POINT_SPEED, 2)
    })


def init_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video device.")
    return cap


def process_frame(frame):
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, rgb_frame


def draw_landmarks(frame, hands, drawing_utils):
    for hand in hands:
        drawing_utils.draw_landmarks(frame, hand)
        landmarks = hand.landmark
        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)
        for connection in mp.solutions.hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_x = int(landmarks[start_idx].x * frame.shape[1])
            start_y = int(landmarks[start_idx].y * frame.shape[0])
            end_x = int(landmarks[end_idx].x * frame.shape[1])
            end_y = int(landmarks[end_idx].y * frame.shape[0])
            cv2.line(frame, (start_x, start_y), (end_x, end_y), LINE_COLOR, LINE_THICKNESS)
    return landmarks


def get_landmark_coordinates(landmarks, frame_width, frame_height):
    coords = {}
    for id, landmark in enumerate(landmarks):
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        coords[id] = (x, y)
    return coords


def map_to_screen(coords, screen_width, screen_height, frame_width, frame_height):
    mapped_coords = {}
    for id, (x, y) in coords.items():
        mapped_x = screen_width * x / frame_width
        mapped_y = screen_height * y / frame_height
        mapped_coords[id] = (mapped_x, mapped_y)
    return mapped_coords


def move_cursor(index_coords, plocx, plocy, smoothening):
    index_x, index_y = index_coords
    clocx = plocx + (index_x - plocx) / smoothening
    clocy = plocy + (index_y - plocy) / smoothening

    # Apply scaling factor
    clocx = plocx + (clocx - plocx) * SCALING_FACTOR
    clocy = plocy + (clocy - plocy) * SCALING_FACTOR

    pyautogui.moveTo(clocx, clocy)
    return clocx, clocy


def detect_gestures(coords, thumb_coords, click_time, click_threshold, single_click_flag, left_dragging):
    thumb_x, thumb_y = thumb_coords

    if abs(coords[8][1] - thumb_y) < 70:
        current_time = time.time()
        if current_time - click_time < click_threshold:
            pyautogui.doubleClick()
            click_time = 0
        else:
            if not single_click_flag:
                pyautogui.click()
                single_click_flag = True
            click_time = current_time
    else:
        single_click_flag = False

    if abs(coords[16][1] - thumb_y) < 70:
        if not left_dragging:
            pyautogui.mouseDown()
            left_dragging = True
    else:
        if left_dragging:
            pyautogui.mouseUp()
            left_dragging = False

    if abs(coords[12][1] - thumb_y) < 70:
        pyautogui.rightClick()
        time.sleep(1)

    extended_fingers = [id for id in [8, 12, 16, 20] if coords[id][1] < coords[id - 2][1]]

    if len(extended_fingers) == 0:
        thumb_tip_y = coords[4][1]
        wrist_y = coords[0][1]

        if thumb_tip_y < wrist_y - 40:
            pyautogui.scroll(200)
        elif thumb_tip_y > wrist_y + 40:
            pyautogui.scroll(-200)

    return click_time, single_click_flag, left_dragging


def add_user_instructions(frame):
    overlay = frame.copy()
    alpha = 0.6
    cv2.rectangle(overlay, (5, 5), (450, 200), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    instructions = [
        "Virtual Mouse Instructions:",
        "1. Move cursor: Use Index Finger",
        "2. Left Click: Thumb + Index Finger",
        "3. Right Click: Thumb + Middle Finger",
        "4. Drag: Thumb + Ring Finger",
        "5. Scroll: Thumbs Up / Down"
    ]
    y0, dy = 30, 30
    for i, line in enumerate(instructions):
        y = y0 + i * dy
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def update_points(frame_width, frame_height):
    for point in points:
        point['pos'] += point['vel'] / np.array([frame_width, frame_height])

        if point['pos'][0] <= 0 or point['pos'][0] >= 1:
            point['vel'][0] *= -1
        if point['pos'][1] <= 0 or point['pos'][1] >= 1:
            point['vel'][1] *= -1


def draw_mesh(frame):
    frame_height, frame_width, _ = frame.shape
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            pi = points[i]['pos'] * np.array([frame_width, frame_height])
            pj = points[j]['pos'] * np.array([frame_width, frame_height])
            dist = np.linalg.norm(pi - pj)
            if dist < CONNECT_DIST:
                intensity = 1 - (dist / CONNECT_DIST)
                color = (int(255 * intensity), int(255 * intensity), 255)
                cv2.line(frame, tuple(pi.astype(int)), tuple(pj.astype(int)), color, 1)


def main():
    cap = init_webcam()
    hand_detector = mp.solutions.hands.Hands()
    drawing_utils = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()
    smoothening = 7
    plocx, plocy = 0, 0
    click_time = 0
    click_threshold = 0.3
    single_click_flag = False
    left_dragging = False

    cv2.namedWindow('Virtual Mouse', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame, rgb_frame = process_frame(frame)
        frame_height, frame_width, _ = frame.shape
        update_points(frame_width, frame_height)

        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        if hands:
            hand = hands[0]
            landmarks = draw_landmarks(frame, [hand], drawing_utils)
            coords = get_landmark_coordinates(landmarks, frame_width, frame_height)
            mapped_coords = map_to_screen(coords, screen_width, screen_height, frame_width, frame_height)
            clocx, clocy = move_cursor(mapped_coords[8], plocx, plocy, smoothening)
            plocx, plocy = clocx, clocy

            click_time, single_click_flag, left_dragging = detect_gestures(
                mapped_coords, mapped_coords[4], click_time, click_threshold, single_click_flag, left_dragging
            )

        draw_mesh(frame)
        frame = add_user_instructions(frame)

        new_width = cv2.getWindowImageRect('Virtual Mouse')[2]
        new_height = cv2.getWindowImageRect('Virtual Mouse')[3]

        if new_width > 0 and new_height > 0:
            frame = cv2.resize(frame, (new_width, new_height))

        cv2.imshow('Virtual Mouse', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


'''



'''
import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import random

# Constants for drawing
CIRCLE_RADIUS = 5
CIRCLE_COLOR = (255, 255, 0)  # Neon Sky Blue
CIRCLE_THICKNESS = -1  # Filled circle
LINE_COLOR = (255, 255, 0)
LINE_THICKNESS = 2
SCALING_FACTOR = 5.0

# Mesh settings
NUM_POINTS = 25  # Light mesh
POINT_SPEED = 0.5
CONNECT_DIST = 150

# Initialize points for moving mesh
points = []
for _ in range(NUM_POINTS):
    points.append({
        "pos": np.array([random.uniform(0, 1), random.uniform(0, 1)]),
        "vel": np.random.uniform(-POINT_SPEED, POINT_SPEED, 2)
    })

def init_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video device.")
    return cap

def process_frame(frame):
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, rgb_frame

def draw_landmarks(frame, hands, drawing_utils):
    for hand in hands:
        drawing_utils.draw_landmarks(frame, hand)
        landmarks = hand.landmark
        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)
        for connection in mp.solutions.hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_x = int(landmarks[start_idx].x * frame.shape[1])
            start_y = int(landmarks[start_idx].y * frame.shape[0])
            end_x = int(landmarks[end_idx].x * frame.shape[1])
            end_y = int(landmarks[end_idx].y * frame.shape[0])
            cv2.line(frame, (start_x, start_y), (end_x, end_y), LINE_COLOR, LINE_THICKNESS)
    return landmarks

def get_landmark_coordinates(landmarks, frame_width, frame_height):
    coords = {}
    for id, landmark in enumerate(landmarks):
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        coords[id] = (x, y)
    return coords

def map_to_screen(coords, screen_width, screen_height, frame_width, frame_height):
    mapped_coords = {}
    for id, (x, y) in coords.items():
        mapped_x = screen_width * x / frame_width
        mapped_y = screen_height * y / frame_height
        mapped_coords[id] = (mapped_x, mapped_y)
    return mapped_coords

def move_cursor(index_coords, plocx, plocy, smoothening):
    index_x, index_y = index_coords
    clocx = plocx + (index_x - plocx) / smoothening
    clocy = plocy + (index_y - plocy) / smoothening

    # Apply scaling factor
    clocx = plocx + (clocx - plocx) * SCALING_FACTOR
    clocy = plocy + (clocy - plocy) * SCALING_FACTOR

    pyautogui.moveTo(clocx, clocy)
    return clocx, clocy

def detect_gestures(coords, thumb_coords, click_time, click_threshold, single_click_flag, left_dragging):
    thumb_x, thumb_y = thumb_coords

    if abs(coords[8][1] - thumb_y) < 70:
        current_time = time.time()
        if current_time - click_time < click_threshold:
            pyautogui.doubleClick()
            click_time = 0
        else:
            if not single_click_flag:
                pyautogui.click()
                single_click_flag = True
            click_time = current_time
    else:
        single_click_flag = False

    if abs(coords[16][1] - thumb_y) < 70:
        if not left_dragging:
            pyautogui.mouseDown()
            left_dragging = True
    else:
        if left_dragging:
            pyautogui.mouseUp()
            left_dragging = False

    if abs(coords[12][1] - thumb_y) < 70:
        pyautogui.rightClick()
        time.sleep(1)

    extended_fingers = [id for id in [8, 12, 16, 20] if coords[id][1] < coords[id - 2][1]]

    if len(extended_fingers) == 0:
        thumb_tip_y = coords[4][1]
        wrist_y = coords[0][1]

        if thumb_tip_y < wrist_y - 40:
            pyautogui.scroll(200)
        elif thumb_tip_y > wrist_y + 40:
            pyautogui.scroll(-200)

    return click_time, single_click_flag, left_dragging

def add_user_instructions(frame):
    overlay = frame.copy()
    alpha = 0.6
    cv2.rectangle(overlay, (5, 5), (450, 200), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    instructions = [
        "Virtual Mouse Instructions:",
        "1. Move cursor: Use Index Finger",
        "2. Left Click: Thumb + Index Finger",
        "3. Right Click: Thumb + Middle Finger",
        "4. Drag: Thumb + Ring Finger",
        "5. Scroll: Thumbs Up / Down"
    ]
    y0, dy = 30, 30
    for i, line in enumerate(instructions):
        y = y0 + i * dy
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return frame

def update_points(frame_width, frame_height):
    for point in points:
        point['pos'] += point['vel'] / np.array([frame_width, frame_height])

        if point['pos'][0] <= 0 or point['pos'][0] >= 1:
            point['vel'][0] *= -1
        if point['pos'][1] <= 0 or point['pos'][1] >= 1:
            point['vel'][1] *= -1

def draw_mesh(frame):
    frame_height, frame_width, _ = frame.shape
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            pi = points[i]['pos'] * np.array([frame_width, frame_height])
            pj = points[j]['pos'] * np.array([frame_width, frame_height])
            dist = np.linalg.norm(pi - pj)
            if dist < CONNECT_DIST:
                intensity = 1 - (dist / CONNECT_DIST)
                color = (int(255 * intensity), int(255 * intensity), 255)
                cv2.line(frame, tuple(pi.astype(int)), tuple(pj.astype(int)), color, 1)

def main():
    cap = init_webcam()
    hand_detector = mp.solutions.hands.Hands()
    drawing_utils = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()
    smoothening = 7
    plocx, plocy = 0, 0
    click_time = 0
    click_threshold = 0.3
    single_click_flag = False
    left_dragging = False

    cv2.namedWindow('Virtual Mouse', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Virtual Mouse', 1280, 720)  # Set your desired fixed size

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame, rgb_frame = process_frame(frame)
        frame_height, frame_width, _ = frame.shape
        update_points(frame_width, frame_height)

        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        if hands:
            hand = hands[0]
            landmarks = draw_landmarks(frame, [hand], drawing_utils)
            coords = get_landmark_coordinates(landmarks, frame_width, frame_height)
            mapped_coords = map_to_screen(coords, screen_width, screen_height, frame_width, frame_height)
            clocx, clocy = move_cursor(mapped_coords[8], plocx, plocy, smoothening)
            plocx, plocy = clocx, clocy

            click_time, single_click_flag, left_dragging = detect_gestures(
                mapped_coords, mapped_coords[4], click_time, click_threshold, single_click_flag, left_dragging
            )

        draw_mesh(frame)
        frame = add_user_instructions(frame)

        cv2.imshow('Virtual Mouse', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

'''











'''
import cv2
import mediapipe as mp
import time
import numpy as np
import random

# Constants for drawing
CIRCLE_RADIUS = 5
CIRCLE_COLOR = (255, 255, 0)  # Neon Sky Blue
CIRCLE_THICKNESS = -1
LINE_COLOR = (255, 255, 0)
LINE_THICKNESS = 2
SCALING_FACTOR = 5.0

# Mesh settings
NUM_POINTS = 25
POINT_SPEED = 0.5
CONNECT_DIST = 150

# Initialize points for moving mesh
points = []
for _ in range(NUM_POINTS):
    points.append({
        "pos": np.array([random.uniform(0, 1), random.uniform(0, 1)]),
        "vel": np.random.uniform(-POINT_SPEED, POINT_SPEED, 2)
    })

def init_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video device.")
    return cap

def process_frame(frame):
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, rgb_frame

def draw_landmarks(frame, hands, drawing_utils):
    for hand in hands:
        drawing_utils.draw_landmarks(frame, hand)
        landmarks = hand.landmark
        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)
        for connection in mp.solutions.hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_x = int(landmarks[start_idx].x * frame.shape[1])
            start_y = int(landmarks[start_idx].y * frame.shape[0])
            end_x = int(landmarks[end_idx].x * frame.shape[1])
            end_y = int(landmarks[end_idx].y * frame.shape[0])
            cv2.line(frame, (start_x, start_y), (end_x, end_y), LINE_COLOR, LINE_THICKNESS)
    return landmarks

def get_landmark_coordinates(landmarks, frame_width, frame_height):
    coords = {}
    for id, landmark in enumerate(landmarks):
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        coords[id] = (x, y)
    return coords

def add_user_instructions(frame):
    overlay = frame.copy()
    alpha = 0.6
    cv2.rectangle(overlay, (5, 5), (450, 200), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    instructions = [
        "Virtual Mouse (Demo Mode - No Control):",
        "1. Move hand to see finger tracking",
        "2. Cursor/mouse features removed",
        "3. Still showing animated mesh",
        "4. Visual hand detection only"
    ]
    y0, dy = 30, 30
    for i, line in enumerate(instructions):
        y = y0 + i * dy
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return frame

def update_points(frame_width, frame_height):
    for point in points:
        point['pos'] += point['vel'] / np.array([frame_width, frame_height])
        if point['pos'][0] <= 0 or point['pos'][0] >= 1:
            point['vel'][0] *= -1
        if point['pos'][1] <= 0 or point['pos'][1] >= 1:
            point['vel'][1] *= -1

def draw_mesh(frame):
    frame_height, frame_width, _ = frame.shape
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            pi = points[i]['pos'] * np.array([frame_width, frame_height])
            pj = points[j]['pos'] * np.array([frame_width, frame_height])
            dist = np.linalg.norm(pi - pj)
            if dist < CONNECT_DIST:
                intensity = 1 - (dist / CONNECT_DIST)
                color = (int(255 * intensity), int(255 * intensity), 255)
                cv2.line(frame, tuple(pi.astype(int)), tuple(pj.astype(int)), color, 1)

def main():
    cap = init_webcam()
    hand_detector = mp.solutions.hands.Hands()
    drawing_utils = mp.solutions.drawing_utils

    cv2.namedWindow('Virtual Mouse', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Virtual Mouse', 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame, rgb_frame = process_frame(frame)
        frame_height, frame_width, _ = frame.shape
        update_points(frame_width, frame_height)

        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        if hands:
            hand = hands[0]
            landmarks = draw_landmarks(frame, [hand], drawing_utils)
            coords = get_landmark_coordinates(landmarks, frame_width, frame_height)
            # Optionally: draw fingertip positions or label them

        draw_mesh(frame)
        frame = add_user_instructions(frame)

        cv2.imshow('Virtual Mouse', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
'''


'''
import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import random

# Constants
CIRCLE_RADIUS = 5
CIRCLE_COLOR = (255, 255, 0)  # Neon Sky Blue
CIRCLE_THICKNESS = -1
LINE_COLOR = (255, 255, 0)
LINE_THICKNESS = 2

# Mesh settings
NUM_POINTS = 25
POINT_SPEED = 0.5
CONNECT_DIST = 150

# Initialize moving mesh points
points = []
for _ in range(NUM_POINTS):
    points.append({
        "pos": np.array([random.uniform(0, 1), random.uniform(0, 1)]),
        "vel": np.random.uniform(-POINT_SPEED, POINT_SPEED, 2)
    })

def init_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video device.")
    return cap

def process_frame(frame):
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, rgb_frame

def draw_landmarks(frame, hands, drawing_utils):
    for hand in hands:
        drawing_utils.draw_landmarks(frame, hand)
        landmarks = hand.landmark
        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)
        for connection in mp.solutions.hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_x = int(landmarks[start_idx].x * frame.shape[1])
            start_y = int(landmarks[start_idx].y * frame.shape[0])
            end_x = int(landmarks[end_idx].x * frame.shape[1])
            end_y = int(landmarks[end_idx].y * frame.shape[0])
            cv2.line(frame, (start_x, start_y), (end_x, end_y), LINE_COLOR, LINE_THICKNESS)
    return landmarks

def get_landmark_coordinates(landmarks, frame_width, frame_height):
    coords = {}
    for id, landmark in enumerate(landmarks):
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        coords[id] = (x, y)
    return coords

def map_to_screen(coords, screen_width, screen_height, frame_width, frame_height):
    mapped_coords = {}
    for id, (x, y) in coords.items():
        mapped_x = screen_width * x / frame_width
        mapped_y = screen_height * y / frame_height
        mapped_coords[id] = (mapped_x, mapped_y)
    return mapped_coords

def move_cursor(index_coords, plocx, plocy, smoothening):
    index_x, index_y = index_coords
    clocx = plocx + (index_x - plocx) / smoothening
    clocy = plocy + (index_y - plocy) / smoothening
    pyautogui.moveTo(clocx, clocy)
    return clocx, clocy

def detect_gestures(coords, thumb_coords, click_time, click_threshold, single_click_flag, left_dragging):
    thumb_x, thumb_y = thumb_coords

    if abs(coords[8][1] - thumb_y) < 70:
        current_time = time.time()
        if current_time - click_time < click_threshold:
            pyautogui.doubleClick()
            click_time = 0
        else:
            if not single_click_flag:
                pyautogui.click()
                single_click_flag = True
            click_time = current_time
    else:
        single_click_flag = False

    if abs(coords[16][1] - thumb_y) < 70:
        if not left_dragging:
            pyautogui.mouseDown()
            left_dragging = True
    else:
        if left_dragging:
            pyautogui.mouseUp()
            left_dragging = False

    if abs(coords[12][1] - thumb_y) < 70:
        pyautogui.rightClick()
        time.sleep(1)

    extended_fingers = [id for id in [8, 12, 16, 20] if coords[id][1] < coords[id - 2][1]]

    if len(extended_fingers) == 0:
        thumb_tip_y = coords[4][1]
        wrist_y = coords[0][1]
        if thumb_tip_y < wrist_y - 40:
            pyautogui.scroll(200)
        elif thumb_tip_y > wrist_y + 40:
            pyautogui.scroll(-200)

    return click_time, single_click_flag, left_dragging

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

def add_user_instructions(frame):
    overlay = frame.copy()
    alpha = 0.6
    cv2.rectangle(overlay, (5, 5), (450, 200), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    instructions = [
        "Virtual Mouse Instructions:",
        "1. Move cursor: Use Index Finger",
        "2. Left Click: Thumb + Index Finger",
        "3. Right Click: Thumb + Middle Finger",
        "4. Drag: Thumb + Ring Finger",
        "5. Scroll: Thumbs Up / Down"
    ]

    # Convert the frame to a PIL Image for drawing with Pillow
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    try:
        # Load Futura font
        font = ImageFont.truetype("futura.ttf", 20)  # Replace with the path to the Futura font file
    except IOError:
        # If Futura is not available, use the default PIL font
        font = ImageFont.load_default()

    y0, dy = 30, 30
    for i, line in enumerate(instructions):
        y = y0 + i * dy
        draw.text((15, y), line, font=font, fill=(255, 255, 255))  # White text

    # Convert back to OpenCV format
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return frame


def update_points(frame_width, frame_height):
    for point in points:
        point['pos'] += point['vel'] / np.array([frame_width, frame_height])
        if point['pos'][0] <= 0 or point['pos'][0] >= 1:
            point['vel'][0] *= -1
        if point['pos'][1] <= 0 or point['pos'][1] >= 1:
            point['vel'][1] *= -1

def draw_mesh(frame):
    frame_height, frame_width, _ = frame.shape
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            pi = points[i]['pos'] * np.array([frame_width, frame_height])
            pj = points[j]['pos'] * np.array([frame_width, frame_height])
            dist = np.linalg.norm(pi - pj)
            if dist < CONNECT_DIST:
                intensity = 1 - (dist / CONNECT_DIST)
                color = (255, 0, int(255 * intensity))  # Neon purple
                cv2.line(frame, tuple(pi.astype(int)), tuple(pj.astype(int)), color, 1)

def main():
    cap = init_webcam()
    hand_detector = mp.solutions.hands.Hands()
    drawing_utils = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()
    smoothening = 2  # Reduced smoothing = faster movement
    plocx, plocy = 0, 0
    click_time = 0
    click_threshold = 0.3
    single_click_flag = False
    left_dragging = False

    cv2.namedWindow('Virtual Mouse', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Virtual Mouse', 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame, rgb_frame = process_frame(frame)
        frame_height, frame_width, _ = frame.shape
        update_points(frame_width, frame_height)

        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        if hands:
            hand = hands[0]
            landmarks = draw_landmarks(frame, [hand], drawing_utils)
            coords = get_landmark_coordinates(landmarks, frame_width, frame_height)
            mapped_coords = map_to_screen(coords, screen_width, screen_height, frame_width, frame_height)
            clocx, clocy = move_cursor(mapped_coords[8], plocx, plocy, smoothening)
            plocx, plocy = clocx, clocy

            click_time, single_click_flag, left_dragging = detect_gestures(
                mapped_coords, mapped_coords[4], click_time, click_threshold, single_click_flag, left_dragging
            )

        draw_mesh(frame)
        frame = add_user_instructions(frame)
        cv2.imshow('Virtual Mouse', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

''' 

import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

# Constants
CIRCLE_RADIUS = 5
CIRCLE_COLOR = (255, 255, 0)  # Neon Sky Blue
CIRCLE_THICKNESS = -1
LINE_COLOR = (255, 255, 0)
LINE_THICKNESS = 2

# Mesh settings
NUM_POINTS = 25
POINT_SPEED = 0.5
CONNECT_DIST = 150

# Initialize moving mesh points
points = []
for _ in range(NUM_POINTS):
    points.append({
        "pos": np.array([random.uniform(0, 1), random.uniform(0, 1)]),
        "vel": np.random.uniform(-POINT_SPEED, POINT_SPEED, 2)
    })

def init_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video device.")
    return cap

def process_frame(frame):
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, rgb_frame

def draw_landmarks(frame, hands, drawing_utils):
    for hand in hands:
        drawing_utils.draw_landmarks(frame, hand)
        landmarks = hand.landmark
        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)
        for connection in mp.solutions.hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_x = int(landmarks[start_idx].x * frame.shape[1])
            start_y = int(landmarks[start_idx].y * frame.shape[0])
            end_x = int(landmarks[end_idx].x * frame.shape[1])
            end_y = int(landmarks[end_idx].y * frame.shape[0])
            cv2.line(frame, (start_x, start_y), (end_x, end_y), LINE_COLOR, LINE_THICKNESS)
    return landmarks

def get_landmark_coordinates(landmarks, frame_width, frame_height):
    coords = {}
    for id, landmark in enumerate(landmarks):
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        coords[id] = (x, y)
    return coords

def map_to_screen(coords, screen_width, screen_height, frame_width, frame_height):
    mapped_coords = {}
    for id, (x, y) in coords.items():
        mapped_x = screen_width * x / frame_width
        mapped_y = screen_height * y / frame_height
        mapped_coords[id] = (mapped_x, mapped_y)
    return mapped_coords

def move_cursor(index_coords, plocx, plocy, smoothening):
    index_x, index_y = index_coords
    clocx = plocx + (index_x - plocx) / smoothening
    clocy = plocy + (index_y - plocy) / smoothening
    pyautogui.moveTo(clocx, clocy)
    return clocx, clocy

def detect_gestures(coords, thumb_coords, click_time, click_threshold, single_click_flag, left_dragging):
    thumb_x, thumb_y = thumb_coords

    if abs(coords[8][1] - thumb_y) < 70:
        current_time = time.time()
        if current_time - click_time < click_threshold:
            pyautogui.doubleClick()
            click_time = 0
        else:
            if not single_click_flag:
                pyautogui.click()
                single_click_flag = True
            click_time = current_time
    else:
        single_click_flag = False

    if abs(coords[16][1] - thumb_y) < 70:
        if not left_dragging:
            pyautogui.mouseDown()
            left_dragging = True
    else:
        if left_dragging:
            pyautogui.mouseUp()
            left_dragging = False

    if abs(coords[12][1] - thumb_y) < 70:
        pyautogui.rightClick()
        time.sleep(1)

    extended_fingers = [id for id in [8, 12, 16, 20] if coords[id][1] < coords[id - 2][1]]

    if len(extended_fingers) == 0:
        thumb_tip_y = coords[4][1]
        wrist_y = coords[0][1]
        if thumb_tip_y < wrist_y - 40:
            pyautogui.scroll(200)
        elif thumb_tip_y > wrist_y + 40:
            pyautogui.scroll(-200)

    return click_time, single_click_flag, left_dragging

def add_user_instructions(frame):
    overlay = frame.copy()
    alpha = 0.6
    cv2.rectangle(overlay, (5, 5), (450, 200), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    instructions = [
        "Virtual Mouse Instructions:",
        "1. Move cursor: Use Index Finger",
        "2. Left Click: Thumb + Index Finger",
        "3. Right Click: Thumb + Middle Finger",
        "4. Drag: Thumb + Ring Finger",
        "5. Scroll: Thumbs Up / Down"
    ]

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("futura.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    y0, dy = 30, 30
    for i, line in enumerate(instructions):
        y = y0 + i * dy
        draw.text((15, y), line, font=font, fill=(255, 255, 255))

    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame

def update_points(frame_width, frame_height):
    for point in points:
        point['pos'] += point['vel'] / np.array([frame_width, frame_height])
        if point['pos'][0] <= 0 or point['pos'][0] >= 1:
            point['vel'][0] *= -1
        if point['pos'][1] <= 0 or point['pos'][1] >= 1:
            point['vel'][1] *= -1

def draw_mesh(frame):
    frame_height, frame_width, _ = frame.shape
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            pi = points[i]['pos'] * np.array([frame_width, frame_height])
            pj = points[j]['pos'] * np.array([frame_width, frame_height])
            dist = np.linalg.norm(pi - pj)
            if dist < CONNECT_DIST:
                intensity = 1 - (dist / CONNECT_DIST)
                color = (255, 0, int(255 * intensity))  # Neon purple
                cv2.line(frame, tuple(pi.astype(int)), tuple(pj.astype(int)), color, 1)

def main():
    cap = init_webcam()
    hand_detector = mp.solutions.hands.Hands()
    drawing_utils = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()
    smoothening = 2
    plocx, plocy = 0, 0
    click_time = 0
    click_threshold = 0.3
    single_click_flag = False
    left_dragging = False
    prev_time = 0

    cv2.namedWindow('Virtual Mouse', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Virtual Mouse', 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame, rgb_frame = process_frame(frame)
        frame_height, frame_width, _ = frame.shape
        update_points(frame_width, frame_height)

        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        if hands:
            hand = hands[0]
            landmarks = draw_landmarks(frame, [hand], drawing_utils)
            coords = get_landmark_coordinates(landmarks, frame_width, frame_height)
            mapped_coords = map_to_screen(coords, screen_width, screen_height, frame_width, frame_height)
            clocx, clocy = move_cursor(mapped_coords[8], plocx, plocy, smoothening)
            plocx, plocy = clocx, clocy

            click_time, single_click_flag, left_dragging = detect_gestures(
                mapped_coords, mapped_coords[4], click_time, click_threshold, single_click_flag, left_dragging
            )

        draw_mesh(frame)
        frame = add_user_instructions(frame)

        # FPS calculation and display
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Draw FPS at the top-right corner
        cv2.putText(frame, f'FPS: {int(fps)}', (frame_width - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Virtual Mouse', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



