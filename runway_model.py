import torch
import os
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from network.Transformer import Transformer
import runway
from runway.data_types import *
from save_utils import *

@runway.setup(options={"checkpoint" : file(extension=".pth")})
def setup(opts):
    model = Transformer()
    model.load_state_dict(torch.load(opts["checkpoint"]))
    model.eval()

    if torch.cuda.is_available():
        print("GPU Mode")
        model.cuda()
    else:
        print("CPU Mode")
        model.float()

    return model

command_inputs = {"input_image" : image}
command_outputs = {"output_image" : image}

@runway.command("cartoonize_image", inputs=command_inputs, outputs=command_outputs, description="Cartoonize the input image")
def cartoonize_image(model, inputs):

    img = inputs["input_image"].convert("RGB")

    h, w = img.size[0], img.size[1]
    ratio = h * 1.0 / w
    if ratio > 1:
        w = int(h * 1.0/ratio)
    else:
        h = int(w * 1.0/ratio)

    input_image = img.resize((h, w), Image.BICUBIC)
    input_image = np.asarray(input_image)
    # RGB -> BGR
    input_image = input_image[:, :, [2, 1, 0]]
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    # preprocess, (-1, 1)
    input_image = -1 + 2 * input_image 
    if torch.cuda.is_available():
        input_image = Variable(input_image, volatile=True).cuda()
    else:
        input_image = Variable(input_image, volatile=True).float()
    # forward
    output_image = model(input_image)
    output_image = output_image[0]
    # BGR -> RGB
    output_image = output_image[[2, 1, 0], :, :]
    # deprocess, (0, 1)
    output_image = output_image.data.cpu().float() * 0.5 + 0.5
    
    out = save_image(output_image)

    return {"output_image" : out}

if __name__ == "__main__":
    runway.run(model_options={"checkpoint" : "./pretrained_model/Hayao_net_G_float.pth"})
