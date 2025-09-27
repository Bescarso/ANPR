from ultralytics import YOLO
from functions_anpr import *
import supervision as sv
import cv2


def main(route_plate):

    model_plate = YOLO("models/anpr_l_100.pt")
    model_ocr = YOLO("models/ocr_l_100_AdamW.pt")


    image = cv2.imread(route_plate)

    results = model_plate(image, conf=0.6,verbose=False, save=False)[0]

    detections_plate = sv.Detections.from_ultralytics(results)

    detection_model = detections_plate.xyxy

    coordinates = get_values_from_tensor(detection_model)

    coordinates_agroup = put_four_values_together(coordinates)

    cropped_image = extract_data_from_image(coordinates_agroup, edge=10, image = image)

    rotated_image = rotate_image(cropped_image)

    plate_image = ocr_processing_2(model_ocr, rotated_image)

    for img in plate_image:
        cv2.imshow("Image plate", img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()


if __name__ == "__main__":
    flag = True

    while flag:
        number = input("Choose image test (1 - 30): ")

        try:
            numer_file = int(number)

            if 1 <= numer_file <= 30:
                main(f'img/img_test_{numer_file}.jpeg')
                flag = False
            else:
                print("❌ Error: Please choose a number between 1 and 30.")

        except ValueError:
            print("❌ Error: Please enter a valid numeric value.")
