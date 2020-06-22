from keras.models import load_model
from tkinter import *
import PIL
from PIL import Image, ImageDraw

import os
import cv2

import numpy as np

model = load_model('bajju.h5')
print('Trained weights loaded')

def save():
    #global image_number
    filename = 'test.jpg'  # image_number increments by 1 at every save
    image1.save(filename)
    image = cv2.imread(filename)
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow('binarized image', thresh)
    #cv2.imshow('yo',image)
    _,contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 
    cv2.imshow('Contours', image) 
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)        
        # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = thresh[y:y+h, x:x+w]        
        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18,18))        
        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)        
        # Adding the preprocessed digit to the list of preprocessed digits
        preprocessed_image = (padded_digit)
    
    #cv2.imshow('digit',preprocessed_image)    
    #print(preprocessed_image.shape)
    img = preprocessed_image.reshape(1, 28, 28, 1)
    img = img/255.0
    #predicting the digit
    result = model.predict([img])[0]
    os.remove(filename)
    digit , acc = np.argmax(result), max(result)
   
    print(digit)
    print("Accuracy -- "+str(int(acc*100))+'%')    
    root.destroy()
    
    
    
    
def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y


def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    #r=8
    #cv.create_oval(x-r, y-r, x + r, y + r, fill='black')
    cv.create_oval((lastx, lasty, x, y), width=5)
    #  --- PIL
    #draw.create_oval
    draw.line((lastx, lasty, x, y), fill='black', width=15)
    lastx, lasty = x, y


root = Tk()

lastx, lasty = None, None
image_number = 0

cv = Canvas(root, width=640, height=480, bg='white')
# --- PIL
image1 = PIL.Image.new('RGB', (640, 480), 'white')
draw = ImageDraw.Draw(image1)

cv.bind('<1>', activate_paint)
cv.pack(expand=YES, fill=BOTH)

btn_save = Button(text="predict", command=save)
btn_save.pack()

root.mainloop()
