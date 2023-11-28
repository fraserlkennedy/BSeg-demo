#!/usr/bin/env python
# coding: utf-8


import os
from ultralyticsplus import YOLO, render_result
from PIL import Image



if not os.path.exists('outputs/'):
    os.makedirs('outputs/')



model = YOLO('models/fine-tuned-YOLOv8/best.pt')



def YOLOv8_predict(image_path='data/test_images/image_1.jpeg'):

    prediction = model.predict(image_path)
    
    render = render_result(model=model, image=image_path, result=prediction[0])
    
    n_outputs = str(len(os.listdir('outputs/')))
    
    render.save('outputs/output'+f'{n_outputs}.jpg')

