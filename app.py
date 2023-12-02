import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Create the app
app = tk.Tk()
app.geometry("532x622")
app.title("Stable Diffusion App")
ctk.set_appearance_mode("dark")

# A box to enter prompt
prompt = ctk.CTkEntry(height = 40, width = 512, font = ("Arial", 20), text_color = "black", fg_color = "white", master = None)
prompt.place(x = 10, y = 10)

# Image frame
lmain = ctk.CTkLabel(height=512, width=512, master = None)
lmain.place(x = 10, y = 110)

modelid = "CompVis/stable-diffusion-v1-4" # this is the model the app uses
device = "cpu"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", use_auth_token=auth_token)
pipe.to(device)

def generate():
    """Function that generates image."""
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]

    img = ImageTk.PhotoImage(image)
    image.save("generatedimage.png")
    lmain.configure(image=img)
    

# Button that triggers generation
trigger = ctk.CTkButton(height = 40, width = 120, font = ("Calisto MT", 15), text_color = "white", fg_color = "blue", master = None, command=generate)
trigger.configure(text = "Generate")
trigger.place(x = 206, y = 110)

app.mainloop()