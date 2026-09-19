import cv2
import numpy as np
from mss import MSS
import pytesseract
codenames_board = [
    [("", "") for col in range(5)] for row in range(5)
]
WINDOW_HEIGHT = 730
WINDOW_WIDTH = 1270
CORNER = (330, 190)
COL_PIXELS = WINDOW_WIDTH / 5
ROW_PIXELS = WINDOW_HEIGHT / 5

def get_card_color(card):
    h, w = card.shape[:2]

    margin_x = int(w * 0.2)
    margin_y = int(h * 0.2)

    center = card[margin_y:h-margin_y, margin_x:w-margin_x]

    mean_color = np.mean(center, axis=(0, 1))

    return mean_color

def analyze_card_color(card):
    b, g, r = get_card_color(card)
    #print(f"R: {r} G: {g} B: {b}")
    if(r/g > 1.1 and r/b > 1.2):
        return "N"
    elif(g/r > 1.1 and g/b > 1.4):
        return "G"
    else:
        return "B"

def find_card_title(card):
    h, w = card.shape[:2]
    crop = card[
        int(h * 0.50):int(h * 0.85),
        int(w * 0.10):int(w * 0.90)
    ]
    scale = 8

    large = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )
    text = pytesseract.image_to_string(
        large,
        lang="swe",
        config="--psm 7"
    )
    return text.strip()

def main():
    with MSS() as sct:
        monitor = {
            "left": CORNER[0],
            "top": CORNER[1],
            "width": WINDOW_WIDTH,
            "height": WINDOW_HEIGHT
        }
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:

            x, y, w, h = cv2.boundingRect(contour)

            if w > 60 and h > 40:
                col = round(x / COL_PIXELS)
                row = round(y / ROW_PIXELS)
                card = img[y:y+h, x:x+w]

                codenames_board[row][col] = (analyze_card_color(card), find_card_title(card))
    print(codenames_board)
if __name__ == "__main__":
    main()