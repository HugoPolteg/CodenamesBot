import cv2
import numpy as np
from mss import MSS
import pytesseract
from matplotlib import pyplot as plt
import faiss

from wordfreq import top_n_list
from sentence_transformers import SentenceTransformer

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

def find_card_title(card, lang="eng"):
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
        lang=lang,
        config="--psm 7"
    )
    return text.strip()

def get_board():
    codenames_board = [
        [None for col in range(5)] for row in range(5)
    ]
    black = []
    neutral = []
    green = []
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
                color = analyze_card_color(card)
                title = find_card_title(card)
                match color: 
                    case "B":
                        black.append(title)
                    case "G":
                        green.append(title)
                    case "N":
                        neutral.append(title)
                codenames_board[row][col] = (color, title)
    for row in codenames_board:
        for card in codenames_board:
            if card is None:
                return None

    return codenames_board, black, green, neutral

def get_score(black, green, neutral, alpha_black=-2, alpha_green=2, alpha_neutral=-0.2):
    vocabulary = list(dict.fromkeys(
        word.lower()
        for word in top_n_list("en", 50_000)
        if word.isalpha() and len(word) >= 2
    ))
    green = [g.lower() for g in green]
    black = [b.lower() for b in black]
    neutral = [n.lower() for n in neutral]
    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2"
    )
    word_vectors = model.encode(
        vocabulary,
        normalize_embeddings=True,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")
    green_vectors = model.encode(
        green, normalize_embeddings=True
    )
    black_vectors = model.encode(
        black, 
        normalize_embeddings=True
    )
    neutral_vectors = model.encode(
        neutral, normalize_embeddings=True
    )

    index = faiss.IndexFlatIP(word_vectors.shape[1])
    index.add(word_vectors)

    target = green_vectors.mean(axis=0)
    target /= np.linalg.norm(target)

    _, ids = index.search(
        target.astype("float32").reshape(1, -1), 2000
    )

    candidate_ids = ids[0]
    candidate_vectors = word_vectors[candidate_ids]

    green_sim = candidate_vectors @ green_vectors.T
    black_sim = candidate_vectors @ black_vectors.T
    neutral_sim = candidate_vectors @ neutral_vectors.T

    scores = (
        alpha_green * green_sim.mean(axis=1) + 
        alpha_black * black_sim.mean(axis=1) +
        alpha_neutral * neutral_sim.mean(axis=1)
    )

    excluded = set(black + green + neutral)
    results = sorted(
        [
            (vocabulary[candidate_ids[i]], float(scores[i]))
            for i in range(len(candidate_ids))
            if vocabulary[candidate_ids[i]] not in excluded
        ],
        key=lambda item: item[1],
        reverse=True
    )
    for word, score in results[:20]:
        print(f"{word:20s} {score:.4f}")

def main():
    codenames_board, black, green, neutral = get_board()
    print(codenames_board)
    get_score(black, green, neutral)


if __name__ == "__main__":
    main()