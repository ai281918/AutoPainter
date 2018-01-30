import os
import cv2
import util
import math
import random
import numpy as np
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

###################### Window ######################
# variable
USE_MODEL = False
PADDINGS = [2, 4, 8, 16, 32]
DRAWING_AREA = (512, 512, 3)
SHOW_AREA = (128, 128, 3)
TOOLBOX_AREA = (924, 32)
WINDOW_AREA = (TOOLBOX_AREA[0], TOOLBOX_AREA[1]+DRAWING_AREA[1]+PADDINGS[2]*2)

if USE_MODEL:
    import model_2 as model

# main window
root = Tk()
root.title('Hi')
root.geometry(str(WINDOW_AREA[0])+'x'+str(WINDOW_AREA[1]))
root.resizable(width=False, height=False)
root.columnconfigure(1, weight=1)
root.rowconfigure(1, weight=1)

# drawing area
drawing_panel = Label(root, width=DRAWING_AREA[0], height=DRAWING_AREA[1], borderwidth=2, relief='groove')
drawing_panel.place(x=PADDINGS[1]+PADDINGS[2]+SHOW_AREA[0], y=TOOLBOX_AREA[1]+PADDINGS[2]+PADDINGS[1])

# show area
show_panel_1 = Label(root, width=SHOW_AREA[0], height=SHOW_AREA[1], borderwidth=2, relief='groove')
show_panel_1.place(x=PADDINGS[1], y=TOOLBOX_AREA[1]+PADDINGS[2]+PADDINGS[1])

# palette
palette_panel = Label(root, width=256, height=256, borderwidth=2, relief='groove')
palette_panel.place(x=PADDINGS[1]+PADDINGS[2]*2+SHOW_AREA[0]+DRAWING_AREA[0], y=TOOLBOX_AREA[1]+PADDINGS[2]+PADDINGS[1])

palette_square = Label(root, width=128, height=128, borderwidth=0, relief='groove')
palette_square.place(x=PADDINGS[1]+PADDINGS[2]*2+SHOW_AREA[0]+DRAWING_AREA[0]+65, y=TOOLBOX_AREA[1]+PADDINGS[2]+PADDINGS[1]+65)


# button
stencils = {'pen': 0, 'eraser':1}
def highlight_button(id):
    global stencil_buttons
    for button in stencil_buttons:
        button.configure(background='white')
    stencil_buttons[id].configure(background='gray')

def pen_button_cllick():
    global stencil_color, stencil_id
    stencil_id = stencils['pen']
    stencil_color = (0, 0, 0)
    highlight_button(stencils['pen'])

def eraser_button_click():
    global stencil_color, stencil_id
    stencil_id = stencils['eraser']
    stencil_color = (255, 255, 255)
    highlight_button(stencils['eraser'])

pen_image = PhotoImage(file='resource/pen.png')
pen_button = Button(root, width=32, height=32, image=pen_image, command=pen_button_cllick)
pen_button.place(x=PADDINGS[1]+PADDINGS[2]+SHOW_AREA[0], y=PADDINGS[1])
pen_button.configure(background='gray')
eraser_image = PhotoImage(file='resource/eraser.png')
eraser_button = Button(root, image=eraser_image, width=32, height=32, command=eraser_button_click)
eraser_button.place(x=PADDINGS[1]+PADDINGS[2]*2+SHOW_AREA[0]+32, y=PADDINGS[1])

stencil_buttons = [pen_button, eraser_button]

###################### Drawing ######################
img = np.ones((512, 512, 3), np.uint8) * 255
img_re = np.ones((128, 128, 3), np.uint8) * 255
palette_img = np.ones((256, 256, 3)) * 240
palette_square_image = np.zeros((128, 128, 3))
palette_weighting_1 = np.zeros((128, 128, 3))
palette_weighting_2 = np.zeros((128, 128, 3))
palette_H = np.array([255, 0, 0])
stencil_color = (0, 0, 0)
stencil_size = np.ones(len(stencils), np.int) * 5
stencil_id = 0
mouse_x = mouse_y = pre_x = pre_y = 0
palette_circle_x, palette_circle_y, palette_square_x, palette_square_y = 195, 33, 127, 0
palette_pressing = False
def reImage(img, size=(128,128), gray=False):
    if gray:
        return np.array(Image.fromarray(img).resize(size, Image.ANTIALIAS))[:,:,0] / 255.
    return np.array(Image.fromarray(img).resize(size, Image.ANTIALIAS))/ 255.


def Filtering(img, k=5):
    blur = cv2.GaussianBlur(img,(k,k),0)
    return blur

def draw_line(p1, p2):
    global img, stencil_id
    cv2.line(img,p1,p2,tuple(palette_square_image[palette_square_y][palette_square_x]),stencil_size[stencil_id])

def update_drawing_panel():
    global img, drawing_panel, mouse_x, mouse_y
    img_out = np.copy(img)
    cv2.circle(img_out,(mouse_x,mouse_y),int(stencil_size[stencil_id]/2+1),(0,0,0),1)
    cv2.circle(img_out,(mouse_x,mouse_y),int(stencil_size[stencil_id]/2),(255,255,255),1)
    photo = ImageTk.PhotoImage(image=Image.fromarray(img_out))
    drawing_panel.configure(image=photo)
    drawing_panel.image = photo

def update_showing_panel():
    global img_re, show_panel_1
    photo = ImageTk.PhotoImage(image=Image.fromarray(img_re))
    show_panel_1.configure(image=photo)
    show_panel_1.image = photo

def get_circle_color(x, y):
    theta = math.atan2(x, y)
    color = np.zeros(3)
    k = math.pi/3
    if theta >= k*2:
        color[2] = 1
        color[1] = (k*3-theta)/k
    elif theta >= k:
        color[1] = 1
        color[2] = (theta-k)/k
    elif theta >= 0:
        color[1] = 1
        color[0] = (k-theta)/k
    elif theta >= -k:
        color[0] = 1
        color[1] = (theta+k)/k
    elif theta >= -2*k:
        color[0] = 1
        color[2] = (-k-theta)/k
    else:
        color[2] = 1
        color[0] = (theta+3*k)/k
    return color*255

def create_palette():
    global palette_img, palette_weighting_1

    for i in range(128):
        for j in range(128):
            palette_weighting_1[i][j] = (127-j)/127
            palette_weighting_2[i][j] = (i)/127*255

    for i in range(-127, 128):
        for j in range(-127, 128):
            if (np.linalg.norm((i, j)) > 125 and np.linalg.norm((i, j)) < 126.5) or (np.linalg.norm((i, j)) > 104 and np.linalg.norm((i, j)) < 105.5):
                palette_img[i+127][j+127] = 255
            elif np.linalg.norm((i, j)) < 125 and np.linalg.norm((i, j)) > 105.5:
                palette_img[i+127][j+127] = get_circle_color(i, j)
            elif (np.linalg.norm((i, j)) > 126.5 and np.linalg.norm((i, j)) < 128) or (np.linalg.norm((i, j)) > 102.5 and np.linalg.norm((i, j)) < 104):
                palette_img[i+127][j+127] = 200

            if (i== -66 or i == 66) and (j >= -66 and j <= 66):
                palette_img[i+127][j+127] = 150
            if (j== -66 or j == 66) and (i >= -66 and i <= 66):
                palette_img[i+127][j+127] = 150

def update_palette():
    global palette_img, palette_panel, palette_H

    img_out = np.copy(palette_img).astype(np.uint8)
    cv2.circle(img_out,(palette_circle_x,palette_circle_y),4,(0,0,0),2)
    cv2.circle(img_out,(palette_square_x+63,palette_square_y+65),4,(0,0,0),2)
    photo = ImageTk.PhotoImage(image=Image.fromarray(img_out))
    palette_panel.configure(image=photo)
    palette_panel.image = photo

def update_palette_square():
    global palette_square_image, palette_square, palette_weighting_1

    palette_square_image = (-palette_img[palette_circle_y][palette_circle_x]+255)*palette_weighting_1 + palette_img[palette_circle_y][palette_circle_x]
    palette_square_image[palette_square_image>255] = 255
    palette_square_image -= palette_weighting_2
    palette_square_image[palette_square_image<0] = 0

    img_out = np.copy(palette_square_image).astype(np.uint8)
    if max(palette_square_image[palette_square_y][palette_square_x]) > 128:
        cv2.circle(img_out,(palette_square_x,palette_square_y),4,(0,0,0),2)
    else:
        cv2.circle(img_out,(palette_square_x,palette_square_y),4,(256,256,256),2)
    photo = ImageTk.PhotoImage(image=Image.fromarray(img_out))
    palette_square.configure(image=photo)
    palette_square.image = photo

def set_drawing_panel_events():
    global drawing_panel

    def left_button_click(event):
        global pre_x, pre_y
        pre_x = event.x
        pre_y = event.y
        mouse_motion(event)

    def left_button_move(event):
        global pre_x, pre_y
        draw_line((pre_x, pre_y), (event.x, event.y))   
        pre_x = event.x
        pre_y = event.y
        mouse_motion(event)

    def wheel(event):
        global stencil_size
        if event.delta > 0:
            stencil_size[stencil_id] += 2
        else:
            stencil_size[stencil_id] = max(2, stencil_size[stencil_id]-2)

    def mouse_motion(event):
        global mouse_x, mouse_y
        mouse_x, mouse_y = event.x, event.y

    drawing_panel.bind('<Button-1>', left_button_click)
    drawing_panel.bind('<B1-Motion>', left_button_move)
    drawing_panel.bind('<MouseWheel>', wheel)
    drawing_panel.bind('<Motion>', mouse_motion)

def set_palette_event():
    global palette_panel

    def update_palate_position(event):
        global palette_circle_x, palette_circle_y, palette_square_x, palette_square_y, palette_pressing

        if np.linalg.norm((event.x-127, event.y-127)) < 125 and np.linalg.norm((event.x-127, event.y-127)) > 105.5:
            palette_pressing = True

        if not palette_pressing:
            return

        palette_circle_x, palette_circle_y = event.x, event.y
        t = np.linalg.norm((palette_circle_x-127, palette_circle_y-127))
        palette_circle_x = int((palette_circle_x-127) / t * 115) + 127
        palette_circle_y = int((palette_circle_y-127) / t * 115) + 127
        update_palette()
        update_palette_square()
                

    def left_button_click(event):
        update_palate_position(event)

    def left_button_release(event):
        global palette_pressing
        palette_pressing = False
        
    def left_button_move(event):
        update_palate_position(event)
    palette_panel.bind('<Button-1>', left_button_click)
    palette_panel.bind('<B1-Motion>', left_button_move)
    palette_panel.bind('<ButtonRelease-1>', left_button_release)

def set_palette_square_event():
    global palette_square

    def update_palate_position(event):
        global palette_square_x, palette_square_y, palette_pressing

        palette_square_x, palette_square_y = min(max(event.x,0),127), min(max(event.y,0),127)
        update_palette_square()
        update_palette()
                
    def left_button_click(event):
        update_palate_position(event)

        
    def left_button_move(event):
        update_palate_position(event)

    palette_square.bind('<Button-1>', left_button_click)
    palette_square.bind('<B1-Motion>', left_button_move)

def main():
    global img_re

    create_palette()
    update_palette()
    update_palette_square()
    set_palette_event()
    set_drawing_panel_events()
    set_palette_square_event()
    cnt = 0
    while True:
        if cnt > 5:
            if USE_MODEL:
                img_re = reImage(img, gray=True)
                img_re = model.AutoDraw(img_re)*255
                img_re = np.reshape(img_re, [128,128]).astype(np.int8)
                img_re = np.array(Image.fromarray(img_re, 'L').resize((128,128), Image.ANTIALIAS))
            cnt = 0
        cnt += 1

        update_drawing_panel()
        update_showing_panel()
        root.update()
    root.mainloop()

if __name__ == '__main__':
    main()