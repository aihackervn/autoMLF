import argparse
from PIL import Image

from core.autoML.AutoDetect.vietnamese_nlp.vietocr.tool.predictor import Predictor
from core.autoML.AutoDetect.vietnamese_nlp.vietocr.tool.config import Cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='foo help')
    parser.add_argument('--config', required=True, help='foo help')

    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)

    detector = Predictor(config)

    img = Image.open(args.img)
    s = detector.predict(img)

    print(s)

if __name__ == '__main__':
    main()
