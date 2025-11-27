import cv2
from tkinter import Tk, filedialog
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deepface import DeepFace
import ollama

#Kép kiválasztása
def choose_file():

    root = Tk()

    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Válassz egy képet",
        filetypes=[("Kép fájlok", "*.jpg *.jpeg *.png *.bmp")]
    )

    return file_path

#Arc detektálás 
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                         'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    return faces

#Arcok kirajzolása
def draw_faces(image, faces):
    img_copy = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    return img_copy

#Érzelem, kor és nem felismerés DeepFace használatával
def analyze_emotions(image, faces):

    img_copy = image.copy()

    emotions_data = []

    for (x, y, w, h) in faces:
        face_img = image[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        try:
            result = DeepFace.analyze(img_path=face_rgb,
                                      actions=['emotion', 'age', 'gender'],
                                      enforce_detection=False)
            if isinstance(result, list):
                result = result[0]

            emotions = result.get('emotion', {})
            dominant = result.get('dominant_emotion', 'Nem sikerült kiolvasni.')
            age = result.get('age', 'Nem sikerült kiolvasni.')
            gender_data = result.get('gender', {})
            # Extract dominant gender from dictionary
            gender = max(gender_data, key=gender_data.get) if isinstance(gender_data, dict) else str(gender_data)

        except Exception as e:
            emotions = {}
            dominant = "Nem sikerült kiolvasni."
            age = "Nem sikerült kiolvasni."
            gender = "Nem sikerült kiolvasni."
            print("Hiba:", e)

        #kirajzolás
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_copy, f"Domináns: {dominant}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        offset = 20
        for emo, val in emotions.items():
            text = f"{emo}: {val:.2f}"
            cv2.putText(img_copy, text, (x, y+h+offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            offset += 15

        #gender és age kiírás
        cv2.putText(img_copy, f"Gender: {gender}", (x, y+h+offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        offset += 15
        cv2.putText(img_copy, f"Age: {age}", (x, y+h+offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        emotions_data.append({
            "emotions": emotions,
            "dominant": dominant,
            "age": age,
            "gender": gender
        })

    return img_copy, emotions_data


#Emoji generálás ollamaval
def generate_emoji(emotions: dict, dominant: str):
    try:
        response = ollama.chat(
            model="gemma3:1b",
            messages=[
                {"role": "system", "content": "Válaszolj CSAK egyetlen emoji karakterrel. Semmi más."},
                {"role": "user", "content": f"Domináns érzelem: {dominant}. Milyen emoji illik hozzá?"}
            ],
            options={"temperature": 0.3}
        )
        emoji = response['message']['content'].strip()
        return emoji if emoji else "😐"
    except Exception as e:
        print(f"Emoji hiba: {e}")
        return "❓"
    
def create_emoji_image(emoji):
    img = Image.new("RGB", (400, 400), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("seguiemj.ttf", 200)  # Windows emoji font
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), emoji, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    x = (400 - w) / 2
    y = (400 - h) / 2

    draw.text((x, y), emoji, font=font, fill="black")

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)



#Leírás ollamaval 
def generate_description(emotions: dict, dominant: str, age: str, gender: str):
    try:
        response = ollama.chat(
            model="gemma3:1b",
            messages=[
                {"role": "system", "content": "A leírás rövid legyen 1-3 mondat. Egymás után fogod kapni az embereknek az adott érzelmeit. Mindig csak arról az emberről írj akiről az adatokat kapod.  "},
                {"role": "user", "content": f"A következő adatok szerint írj az adott arcról egy leírást:\n"
                                            f"Domináns érzelem: {dominant}\n"
                                            f"Eloszlás: {emotions}\n"
                                            f"Nem: {gender}\n"
                                            f"Kor: {age}"}
            ],
            options={"temperature": 0.5}
        )
        description = response['message']['content']
    except Exception as e:
        description = f"Domináns érzelem: {dominant}, eloszlás: {emotions}, Nem: {gender}, Kor: {age}\n(Hiba: {e})"

    print("\n--- Leírás ---")
    print(description)
    print("--------------------\n")


#állapotok vezérlése    
def pipeline():
    image_path = choose_file()
    if not image_path:
        print("Nem választottál képet.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("Nem sikerült betölteni a képet.")
        return

    faces = detect_faces(image)
    step0 = image.copy()
    step1 = draw_faces(image, faces)
    step2, emotions_data = analyze_emotions(image, faces)

    emoji_images = []
    for i, emo in enumerate(emotions_data, start=1):
        print(f"\n========== Arc {i} ==========")

        emoji = generate_emoji(emo["emotions"], emo["dominant"])
        emoji_images.append(create_emoji_image(emoji))

        print(f"Emoji: {emoji}")
        print(f"Domináns érzelem: {emo['dominant']}")
        print(f"Nem: {emo['gender']}")
        print(f"Kor: {emo['age']}")

        generate_description(
            emo["emotions"],
            emo["dominant"],
            emo["age"],
            emo["gender"]
        )

    steps = [step0, step1, step2]
    titles = ["Eredeti kép", "Arc detektálás", "Érzelem felismerés"]

    step = 0
    emoji_index = 0

    cv2.namedWindow("Pipeline", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Emoji", cv2.WINDOW_NORMAL)

    while True:
        cv2.setWindowTitle("Pipeline", titles[step])
        cv2.imshow("Pipeline", steps[step])

        if emoji_images:
            cv2.imshow("Emoji", emoji_images[emoji_index])

        key = cv2.waitKey(0)

        if key == ord('n'):          
            step = (step + 1) % len(steps)

        elif key == ord('p'):        
            step = (step - 1) % len(steps)

        elif key == ord('e'):       
            emoji_index = (emoji_index + 1) % len(emoji_images)

        elif key == ord('q'):        
            emoji_index = (emoji_index - 1) % len(emoji_images)

        elif key == 27:            
            break

    cv2.destroyAllWindows()
if __name__ == "__main__":
    pipeline()