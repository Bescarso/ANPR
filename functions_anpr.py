from ultralytics import YOLO
import numpy as np
import cv2


def get_values_from_tensor(detection: list = []) -> list:
    data_list = []
    for i in detection:
        i = i.astype(int)
        for r in i:
            data_list.append(r)

    return data_list


def put_four_values_together(data_list: list = []) -> list:
    data_list_four_values = []
    for i in range(0, len(data_list), 4):
        sub_list = data_list[i:i + 4]
        data_list_four_values.append(sub_list)

    return data_list_four_values


def extract_data_from_image(coordinates, edge, image):
    images_plates = []
    for plate in coordinates:
        x1, y1, x2, y2 = plate

        # ROI (Region Of Interest)
        edge = edge
        image_cropped = image[y1 - edge : y2 + edge, x1 - edge :x2+edge]
        images_plates.append(image_cropped)

    return images_plates


def rotate_image (images_cropped):

    images_rotated = []


    for image in images_cropped:
        image_plate = image
        image_gray = cv2.cvtColor(image_plate, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
        valor_maximo = np.argmax(hist)

        # Determinar la relación del valor máximo con respecto a 127.5
        n = 20 if valor_maximo < 100 else 50
        if valor_maximo <100:
            n=28
            m=3
        elif valor_maximo>140:
            n=120
            m=3
        else:
            n=50
            m=4

        threshold = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        kernel = np.ones((m,m), np.uint8)
        closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=9)#3
        edges = cv2.Canny(closing, 50, 180) #50/180

        linesP = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=n, minLineLength=100, maxLineGap=180)

        angles = []


        if linesP is not None:
            for i in range(len(linesP)):
                l = linesP[i][0]
                angle = np.arctan2(l[3] - l[1], l[2] - l[0]) * 180.0 / np.pi
                angles.append(angle)
                #cv2.line(image_plate, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 2) ###

            median_angle = np.median(angles)
            median_angle = np.round(median_angle,2)

        else:
            #print('No se encontró ángulo')
            median_angle = 0
            

        (h, w) = image_plate.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated_image = cv2.warpAffine(image_plate, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        images_rotated.append(rotated_image)
    
    return images_rotated


def ocr_processing_2(model_yolo, rotated_images):
    dict_characters = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
        20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
        30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
    }

    model = YOLO(model_yolo)
    processed_images = []

    for image in rotated_images:
        image_ocr = image.copy()
        results_ocr = model(image_ocr, save=False, verbose=False)[0]
        df_results = results_ocr.to_df()

        if df_results.is_empty():
            print("⚠️ No se detectaron valores en la imagen")
            processed_images.append(image_ocr)
            continue

        letters = df_results["class"].to_list()
        coordinates = df_results["box"].to_list()

        final_results = []
        for i, letter in enumerate(letters):
            coordinate = coordinates[i]
            letra = dict_characters[letter] 
            final_results.append((letra, coordinate['x1']))


        final_results_sorted = sorted(final_results, key=lambda x: x[1])
        plate_text = "".join([letra for letra, _ in final_results_sorted])

        image_ocr = cv2.resize(image_ocr, (640, 480))

        cv2.putText(
            img = image_ocr,
            text = f'Number plate: {plate_text}',
            org = (10, 40),  # plate position
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 1.0,  
            color = (0, 255, 0),  
            thickness=2, 
            lineType=cv2.LINE_AA
        )

        processed_images.append(image_ocr)

    return processed_images
