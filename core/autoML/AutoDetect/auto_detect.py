from autoML.AutoTraining.auto_trainer import save_project
from autoML.AutoDetect.vietnamese_nlp.nlp_onnx import VietnameseOCR
from autoML.AutoDetect.vietnamese_nlp.vietocr.tool.config import Cfg
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import os
import gdown
import base64
import io


def initialize_folder():
    """Initialize the folder for storing weights and models.

    Raises:
        OSError: if the folder cannot be created.
    """
    home = get_model_home()
    OCRHomePath = home + "/.ocr"
    weightsPath = OCRHomePath + "/onnx_model"

    if not os.path.exists(OCRHomePath):
        os.makedirs(OCRHomePath, exist_ok=True)

    if not os.path.exists(weightsPath):
        os.makedirs(weightsPath, exist_ok=True)


def get_model_home():
    """Get the home directory for storing weights and models.

    Returns:
        str: the home directory.
    """
    return str(os.getenv("ITC_AI_HOME", default=str(Path.home())))


def get_ocr_model(url, model_name):
    url_ = url
    initialize_folder()
    home = get_model_home()

    if not os.path.isfile(home + "/.ocr/onnx_model/{}.onnx".format(model_name)):
        output = home + "/.ocr/onnx_model/{}.onnx".format(model_name)
        gdown.download(url_, output, quiet=False)
    model_path = home + "/.ocr/onnx_model/{}.onnx".format(model_name)
    return model_path


def array2base64(image: np.ndarray):
    return base64.b64encode(image)


def process_input_b64(image_b64: base64):
    return base64.b64decode(image_b64)


def process_input(file: bytes) -> np.array:
    return np.asarray(bytearray(io.BytesIO(file).read()), dtype=np.uint8)


def processing_image(file: str) -> np.array:
    return cv2.imdecode(process_input(process_input_b64(file)), cv2.IMREAD_COLOR)


def load_image(image, base64_type=False):
    if base64_type:
        img = processing_image(image)
        img_copy = img.copy()
        return img, img_copy
    else:
        img = cv2.imread(image)
        img_copy = img.copy()
        return img, img_copy


def inference_yolo(image):
    model_path, _ = save_project(delete_model=False)
    model = YOLO(model_path, task='detect')
    result = model.predict(image)
    return result, model, model_path


def detect(image):
    classes, classes_name, bbox, confidences, output_detect = [], [], [], [], []
    image, img_cp = load_image(image)
    result, model, model_path = inference_yolo(img_cp)
    clsname = result[0].boxes.cls
    boxes = result[0].boxes.xyxy
    confidence = result[0].boxes.conf
    for cls, box, confi in zip(clsname, boxes, confidence):
        xmin, ymin, xmax, ymax = map(int, box.tolist())
        bbox.append((xmin, ymin, xmax, ymax))
        classes.append(int(cls))
        classes_name.append(model.names[int(cls)])
        confidences.append(float(confi))
    start_x = [bb[0] for bb in bbox]
    start_y = [bb[1] for bb in bbox]
    end_x = [bb[2] for bb in bbox]
    end_y = [bb[3] for bb in bbox]
    score = [score for score in confidences]
    label_name = [label for label in classes_name]
    return start_x, start_y, end_x, end_y, score, label_name, img_cp


def post_processing(img):
    start_x, start_y, end_x, end_y, score, label_name, img_cp = detect(img)
    detail_ = []
    for xmin, ymin, xmax, ymax, class_name, confidence in zip(start_x,
                                                              start_y,
                                                              end_x,
                                                              end_y,
                                                              label_name,
                                                              score):
        details = {
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'confidence': confidence,
            'class_name': class_name,
        }
        detail_.append(details)

    return detail_, img_cp


def matching(images, task=None, img_show=False):
    output_dir = 'output_detect'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    existing_folders = sum(os.path.isdir(os.path.join(output_dir, folder)) for folder in os.listdir(output_dir))
    new_folder_count = existing_folders + 1
    if task == 'OCR':
        text_results = []
        details, im_cp = post_processing(images)
        vietnamese = VietnameseOCR(
            encode_onnx_model=get_ocr_model(
                url='https://drive.google.com/u/3/uc?id=159JfNNVcHDKrV_00eM_Ve2Elui5nonIc&export=download',
                model_name='encoder'),
            decode_onnx_model=get_ocr_model(
                url='https://drive.google.com/u/3/uc?id=1PUmuoD77dntBT1IWByO6TLNkVLcfH5Db&export=download',
                model_name='decoder'),
            config=Cfg.load_config_from_name('vgg_transformer'))

        cropped_images = [im_cp[d['ymin']:d['ymax'], d['xmin']:d['xmax']] for d in details]
        texts = vietnamese.predict_batch(cropped_images)
        for i, d in enumerate(details):
            xmin = d['xmin']
            xmax = d['xmax']
            ymin = d['ymin']
            ymax = d['ymax']
            confidence = d['confidence']
            class_name = d['class_name']

            result_document = {
                'text': texts[i],
                'bounding_boxes': {
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                },
                'confidence': confidence,
                'class_name': class_name,
            }
            text_results.append(result_document)
            cv2.rectangle(im_cp, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.rectangle(im_cp, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            label = f"{class_name}: {confidence:.2f}"
            text_color = (0, 0, 255)
            text_bg_color = (0, 255, 0)

            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_width, text_height = text_size[0], text_size[1]

            cv2.rectangle(im_cp, (xmin, ymin - text_height - 5), (xmin + text_width, ymin), text_bg_color, -1)

            cv2.putText(im_cp, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        output_folder = os.path.join(output_dir, f'output_folder_{new_folder_count}')
        os.makedirs(output_folder)
        output_path = os.path.join(output_folder, 'output_image.jpg')
        cv2.imwrite(output_path, im_cp)
        if img_show:
            cv2.imshow('OCR', im_cp)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
        print("Result", text_results)
        return text_results
    elif task == 'ObjectDetection':
        detection_detail = []
        details, img_cp = post_processing(images)
        for i, d in enumerate(details):
            xmin = d['xmin']
            xmax = d['xmax']
            ymin = d['ymin']
            ymax = d['ymax']
            confidence = d['confidence']
            class_name = d['class_name']

            result_detection = {
                'bounding_boxes': {
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                },
                'confidence': confidence,
                'class_name': class_name,
            }
            cv2.rectangle(img_cp, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.rectangle(img_cp, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            label = f"{class_name}: {confidence:.2f}"
            text_color = (0, 255, 0)
            text_bg_color = (0, 0, 255)

            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_width, text_height = text_size[0], text_size[1]

            cv2.rectangle(img_cp, (xmin, ymin - text_height - 5), (xmin + text_width, ymin), text_bg_color, -1)

            cv2.putText(img_cp, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            detection_detail.append(result_detection)
        output_folder = os.path.join(output_dir, f'output_folder_{new_folder_count}')
        os.makedirs(output_folder)
        output_path = os.path.join(output_folder, 'output_image.jpg')
        cv2.imwrite(output_path, img_cp)
        if img_show:
            cv2.imshow('ObjectDetection', img_cp)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
        print("Result", detection_detail)
        return detection_detail


if __name__ == '__main__':
    pass
