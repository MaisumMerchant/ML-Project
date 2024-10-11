import cv2
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
# import torch
# from diffusers import AutoPipelineForImage2Image
# from diffusers.utils import make_image_grid, load_image

st.set_page_config(layout="wide")
st.title("Image Processing App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  input_image = Image.open(uploaded_file)
  input_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

  st.subheader("Input Image")
  st.image(input_image, caption="Input Image", use_column_width=True)

  st.subheader("Output Images")

  gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
  st.image(gray, caption="Gray", use_column_width=True)

  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  st.image(blurred, caption="Blurred", use_column_width=True)

  _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
  st.image(thresh, caption="Thresh", use_column_width=True)

  mask = np.zeros_like(thresh)

  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 30]

  for contour in filtered_contours:
    cv2.drawContours(mask, [contour], -1, (255,255,255), thickness=cv2.FILLED)
    cv2.drawContours(input_image, [contour], -1, (255,255,255), thickness=cv2.FILLED)

  st.image(mask, caption="Contours", use_column_width=True)
  st.image(input_image, caption="Contours On Input Image", use_column_width=True)

  stars = np.array([contour[0][0] for contour in filtered_contours])

  if len(stars) >= 3:
    tri = Delaunay(stars)
    lines = stars[tri.simplices]

    for line in lines:
      pt1 = tuple(map(int, line[0]))
      pt2 = tuple(map(int, line[1]))
      cv2.line(input_image, pt1, pt2, (255, 255, 255), 5)

  st.image(input_image, caption="Lines", use_column_width=True)

  image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
  st.image(image_rgb, caption="RGB", use_column_width=True)
  # pipeline = AutoPipelineForImage2Image.from_pretrained(
  #   "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True
  # ).to("cuda")
  # pipeline.enable_model_cpu_offload()

  # prompt = "an animal made from stars"

  # image = pipeline(prompt, image=input_image, num_inference_steps=500).images[0]
  # st.image(image, caption="Output", use_column_width=True)