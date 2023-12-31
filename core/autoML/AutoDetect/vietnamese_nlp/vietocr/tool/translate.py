import cv2
import torch
import numpy as np
import math
from PIL import Image
from torch.nn.functional import log_softmax, softmax
from autoML.AutoDetect.vietnamese_nlp.vietocr.model.transformerocr import VietOCR
from autoML.AutoDetect.vietnamese_nlp.vietocr.model.vocab import Vocab
from autoML.AutoDetect.vietnamese_nlp.vietocr.model.beam import Beam


def batch_translate_beam_search(img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # img: NxCxHxW
    model.eval()
    device = img.device
    sents = []

    with torch.no_grad():
        src = model.cnn(img)
        memories = model.transformer.forward_encoder(src)
        print('all memories: ', memories.shape)
        for i in range(src.size(0)):
            #            memory = memories[:,i,:].repeat(1, beam_size, 1) # TxNxE
            memory = model.transformer.get_memory(memories, i)
            print('memory: ', memory.shape)
            sent = beamsearch(memory, model, device, beam_size, candidates, max_seq_length, sos_token, eos_token)
            sents.append(sent)

    sents = np.asarray(sents)

    return sents


def translate_beam_search(img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # img: 1xCxHxW
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)  # TxNxE
        sent = beamsearch(memory, model, device, beam_size, candidates, max_seq_length, sos_token, eos_token)

    return sent


def beamsearch(memory, model, device, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # memory: Tx1xE
    model.eval()

    beam = Beam(beam_size=beam_size, min_length=0, n_top=candidates, ranker=None, start_token_id=sos_token,
                end_token_id=eos_token)

    with torch.no_grad():
        #        memory = memory.repeat(1, beam_size, 1) # TxNxE
        memory = model.transformer.expand_memory(memory, beam_size)

        for _ in range(max_seq_length):

            tgt_inp = beam.get_current_state().transpose(0, 1).to(device)  # TxN
            decoder_outputs, memory = model.transformer.forward_decoder(tgt_inp, memory)

            log_prob = log_softmax(decoder_outputs[:, -1, :].squeeze(0), dim=-1)
            beam.advance(log_prob.cpu())

            if beam.done():
                break

        scores, ks = beam.sort_finished(minimum=1)

        hypothesises = []
        for i, (times, k) in enumerate(ks[:candidates]):
            hypothesis = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)

    return [1] + [int(i) for i in hypothesises[0][:-1]]


def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    "data: BxCXHxW"
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)
        # print(np.squeeze((memory.detach().cpu().numpy())[: 10]))
        translated_sentence = [[sos_token] * len(img)]
        char_probs = [[1] * len(img)]

        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            # print('Input decoder shape: ', tgt_inp, tgt_inp.shape, memory.shape)
            #           # Main function
            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
            # print(np.squeeze((memory.detach().cpu().numpy())[: 10]))
            output = softmax(output, dim=-1)
            # End main function

            output = output.to('cpu')

            values, indices = torch.topk(output, 5)
            # print('Output decoder shape: ', output.shape, values.shape, indices.shape)

            indices = indices[:, -1, 0]
            indices = indices.tolist()

            values = values[:, -1, 0]
            values = values.tolist()
            char_probs.append(values)

            translated_sentence.append(indices)
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T

        char_probs = np.asarray(char_probs).T
        char_probs = np.multiply(char_probs, translated_sentence > 3)
        char_probs = np.sum(char_probs, axis=-1) / (char_probs > 0).sum(-1)

    return translated_sentence, char_probs


def translate_trt(img, encoder_model, decoder_model, max_seq_length=256, sos_token=1, eos_token=2):
    onnx_inp = {encoder_model.get_inputs()[0].name: img.astype('float32')}
    encoder_output = np.squeeze(encoder_model.run(None, onnx_inp))
    translated_sentence = [[sos_token] * len(img)]
    char_probs = [[1] * len(img)]

    max_length = 0

    while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
        tgt_inp = np.array(translated_sentence).astype('long')
        onnx_decode_inp = {decoder_model.get_inputs()[0].name: tgt_inp.astype('int64'),
                           decoder_model.get_inputs()[1].name: encoder_output.astype('float32')}
        values, indices = decoder_model.run(None, onnx_decode_inp)
        indices = indices[:, -1, 0]
        # indices = np.squeeze(indices)
        indices = indices.tolist()
        values = values[:, -1, 0]
        values = values.tolist()
        char_probs.append(values)
        translated_sentence.append(indices)
        max_length += 1

    translated_sentence = np.asarray(translated_sentence).T
    char_probs = np.asarray(char_probs).T
    line_probs = []
    for i in range(len(img)):
        eos_index = np.where(translated_sentence[i] == eos_token)[0][0]
        line_probs.append(np.mean(char_probs[i][:eos_index]))
    return translated_sentence, line_probs


def build_model(config):
    vocab = Vocab(config['vocab'])
    device = config['device']

    model = VietOCR(len(vocab),
                    config['backbone'],
                    config['cnn'],
                    config['transformer'],
                    config['seq_modeling'])

    model = model.to(device)

    return model, vocab


def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height


def process_image(image, image_height, image_min_width, image_max_width):
    # img = image.convert('RGB')
    img = image[:, :, ::-1]
    h, w = img.shape[:2]
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)
    img = cv2.resize(img, (new_w, image_height))
    img = np.asarray(img).transpose(2, 0, 1)
    img = img / 255
    return img


def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image(image, image_height, image_min_width, image_max_width)
    return np.expand_dims(img, axis=0)


def predict(filename, config):
    img = Image.open(filename)
    img = process_input(img)

    img = img.to(config['device'])

    model, vocab = build_model(config)
    s = translate(img, model)[0].tolist()
    s = vocab.decode(s)

    return s
