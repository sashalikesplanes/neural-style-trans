import streamlit as st
import torchvision.transforms as tfms
#from fastbook import PILImage
from PIL import Image
import numpy as np
import cv2
from torch import tensor
import torch
from main_model import  cnn, cnn_norm_mean, cnn_norm_std, run_style_transfer_with_display, image_loader
import os


device = torch.device('cuda')
unloader = tfms.ToPILImage()

st.title("Image generator")
st.write("Select one of the available style pictures and upload your picture to which apply sthe style")

#uploaded_file = st.file_uploader("Choose your image", type=["jpg", "jpeg", "heic", "png"])

img_size = (1024, 1024)

style_layers = ['conv_1', 'conv_3', 'conv_5', 'conv_9']
content_layers = ['conv_6']


def bytes_to_tns(bytes_str, size):
    nparr = np.fromstring(bytes_str, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_np = cv2.resize(img_np, dsize=size, interpolation=cv2.INTER_CUBIC)
    img_tns = tensor(img_np) / 255
    return img_tns.view(1, 3, size[0], size[1]).clone().to(device, torch.float)


def tns_to_img(tns, size): 
    tns = tns.clone().detach().cpu()
    tns = tns[0].view(size[0], size[1], 3)
    return np.array(tns)

def img_disp(img, run):
    
    img = img.cpu().clone() # clone to not modify the original image
    img = img[0]
    img = unloader(img)
    run_img.image(img, caption=f'Iteration {run}', width=1024)
    


up_file_content = st.file_uploader("Choose a content img")
up_file_style = st.file_uploader("Choose a style img")

run_img = st.empty()

if up_file_content is not None and up_file_style is not None:
    
    # save uploaded images to a file
    cont_file_path = os.path.join("tempDir","cont_file.jpg")
    with open(cont_file_path,"wb") as f: f.write(up_file_content.getbuffer())
    style_file_path = os.path.join("tempDir","style_file.jpg")
    with open(style_file_path,"wb") as f: f.write(up_file_style.getbuffer())
    
    # load the uploaded images from file into tensor for model
    content_img = image_loader(cont_file_path, img_size)
    style_img = image_loader(style_file_path, img_size)
    input_img = image_loader(cont_file_path, img_size)
    # To read file as bytes:
    #content_img = bytes_to_tns(uploaded_file_content.getvalue(), img_size)
    #style_img = bytes_to_tns(uploaded_file_style.getvalue(), img_size)
    #input_img = content_img.clone()
    
    run_style_transfer_with_display(cnn, cnn_norm_mean, cnn_norm_std, content_img, style_img, input_img, content_layers, style_layers, img_disp, num_steps=10000, style_weight=1e8, content_weight=1, iters_to_show=10)