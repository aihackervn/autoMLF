import onnxruntime as rt
import numpy as np

from autoML.AutoDetect.vietnamese_nlp.vietocr.tool.translate import translate_trt, process_input, process_image
from autoML.AutoDetect.vietnamese_nlp.vietocr.model.vocab import Vocab


class VietnameseOCR(object):
    def __init__(self, encode_onnx_model, decode_onnx_model, config):
        session_options = ['CPUExecutionProvider']
        self.encoder_model = rt.InferenceSession(encode_onnx_model, providers=session_options)
        self.decoder_model = rt.InferenceSession(decode_onnx_model, providers=session_options)
        self.config = config
        self.vocab = Vocab(config['vocab'])

    def preprocess_batch(self, list_img):
        """
            list_img: list of PIL Image
        """
        # Get max shape
        batch_width = 0
        batch_list = []
        for idx, img in enumerate(list_img):
            img = process_image(img, self.config['dataset']['image_height'],
                                self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])
            im_width = img.shape[2]
            if im_width > batch_width:
                batch_width = im_width
            batch_list.append(img)
        # Create batch
        batch = np.ones((len(list_img), 3, self.config['dataset']['image_height'], batch_width))
        for idx, single in enumerate(batch_list):
            _, height, width = single.shape
            batch[idx, :, :, :width] = single
        return batch

    def predict(self, img, return_prob=True):
        """
            Predict single-line image
            Input:
                - img: pillow Image
        """
        imgs = process_input(img, self.config['dataset']['image_height'],
                            self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])
        print(imgs.shape)
        s, prob = translate_trt(imgs, self.encoder_model, self.decoder_model)
        s = s[0].tolist()
        prob = prob[0]

        if return_prob:
            return self.vocab.decode(s), prob
        else:
            return self.vocab.decode(s)

    def predict_batch(self, list_img):
        """
            Predict batch of image
            Input:
                - img: pillow Image
        """
        translated_sentence, prob = translate_trt(self.preprocess_batch(list_img), self.encoder_model, self.decoder_model)
        result = []
        for i, s in enumerate(translated_sentence):
            result.append(self.vocab.decode(translated_sentence[i].tolist()))
        return result
