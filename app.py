import gradio as gr
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def caption(img, min_len, max_len):
    raw_image = Image.open(img).convert('RGB')
  
    inputs = processor(raw_image, return_tensors="pt")
    
    out = model.generate(**inputs, min_length=min_len, max_length=max_len)
    return processor.decode(out[0], skip_special_tokens=True)

def greet(img, min_len, max_len):
    start = time.time()
    result = caption(img, min_len, max_len)
    end = time.time()
    total_time = str(end - start)
    result = result + '\n' + total_time + ' seconds'
    return result

iface = gr.Interface(fn=greet, 
                     title='Beetz-Image-Captioning', 
                     description="Task of describing the content of an image in words.",
                     inputs=[gr.Image(type='filepath', label='Image'), gr.Slider(label='Minimum Length', minimum=1, maximum=1000, value=30), gr.Slider(label='Maximum Length', minimum=1, maximum=1000, value=100)], 
                     outputs=gr.Textbox(label='Caption'),
                     theme = gr.themes.Base(primary_hue="teal",secondary_hue="teal",neutral_hue="slate"),)
iface.launch()