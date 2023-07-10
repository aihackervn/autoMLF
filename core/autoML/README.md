# autoML Framework
<div>
<a href="https://console.paperspace.com/github/ultralytics/ultralytics">
    <img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OCR-FF3E1F?style=for-the-badge&logo=ocr&logoColor=white" alt="OCR">
  <img src="https://img.shields.io/badge/Deep_Learning-FF6F00?style=for-the-badge&logo=deep-learning&logoColor=white" alt="Deep Learning">
  <img src="https://img.shields.io/badge/Computer_Vision-5C0099?style=for-the-badge&logo=computer-vision&logoColor=white" alt="Computer Vision">
</div>
<br>

autoMLF is a first release simple tool for training and inference of models on the terminal, with built-in support for OCR and object detection. This `README.md` file provides an overview of the framework and instructions for getting started.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [OCR](#ocr)
  - [Object Detection](#object-detection)
- [Contact](#contact)

## Features

- Training object detection models with ultralytics (yolov8)
- Inference models for applying to OCR or Object Detection 
- OCR (Optical Character Recognition) (Support for English and Vietnamese)
- Object detection

## Software and Hardware requirements:
- All laptop os device
- If you want to run training and inference faster, you might have GPU and CUDA toolkit installed on your device
- Python 3.9+ or Anaconda , project work stable in conda enviroment

# To install autoMLF, follow these steps:
- Quick start
  ```
  pip install automlf
  ```
- For install by cloning repo, please follow step below:
  1. Clone the repository: 

  `git clone https://github.com/autoMLF.git`

  2. Change to the framework directory: 
  
  `cd autoMLF`

  3. Install the dependencies: 
  
  `pip install -r requirements.txt`

## Usage
- Quick Start:
    
  For running tool, please use this command line
  ```
  automls run project
  ```
- If you want to run localy from code, follow it:
  ```
  from autoML.CLI import automltoolkit
  ```
  
## Terminal Transform
- You will see the terminal will suggest you choose some option for training or inference
 ```
Select an option:
1. Training
2. Inference
3. Reset
0. Exit
Enter your choice: 
 ```
- You can choose option for training or inference models

## Checklist

- [x] Simple auto download, create, split data dir
- [x] Apply ultralytics framework into tools for easy to use 
- [x] Adding OCR model for extracting result and recognizing text
- [ ] Add an endpoint demo using FastAPI
- [ ] Add Docker build 
- [ ] Integrate YOLO-NAS for object detection
- [ ] Incorporate Text Detection using DBNet, CRAFT
- [ ] Implement task sharing or collaboration features


## Contact
If you have any questions, suggestions, or feedback, feel free to contact the project maintainer:
- tinprocoder0908@gmail.com
- Linkedin: http://linkedin.com/in/tin-an-22b352271
