import os
import cv2
import util
import math
import queue
import random
import numpy as np
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

class Painter:
    def __init__(self, use_model, rgb, reconstruct_iter, update_freq, input_size, show_area, model):
        self.USE_MODEL = use_model
        self.RGB = rgb
        self.RECONSTRUCT_ITER = reconstruct_iter
        self.update_freq = update_freq
        self.INPUT_SIZE = input_size
        self.SHOW_AREA = show_area
        self.model = model
        PADDINGS = [2, 4, 8, 16, 32]
        DRAWING_AREA = (512, 512, 3)

        if self.RGB:
            TOOLBOX_AREA = (DRAWING_AREA[0]+self.SHOW_AREA[0]+256+26, 32)
        else:
            TOOLBOX_AREA = (DRAWING_AREA[0]+self.SHOW_AREA[0]+26, 32)

        self.WINDOW_AREA = (TOOLBOX_AREA[0], TOOLBOX_AREA[1]+DRAWING_AREA[1]+PADDINGS[2]*2)
        self.save_dir = 'save/'

        # main window
        self.root = Tk()
        self.root.title('Doodle Master')
        self.root.geometry(str(self.WINDOW_AREA[0])+'x'+str(self.WINDOW_AREA[1]))
        self.root.resizable(width=False, height=False)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)

        # drawing area
        self.drawing_panel = Label(self.root, width=DRAWING_AREA[0], height=DRAWING_AREA[1], borderwidth=2, relief='groove')
        self.drawing_panel.place(x=PADDINGS[1]+PADDINGS[2]+self.SHOW_AREA[0], y=TOOLBOX_AREA[1]+PADDINGS[2]+PADDINGS[1])

        # show area
        self.show_panel_1 = Label(self.root, width=self.SHOW_AREA[0], height=self.SHOW_AREA[1], borderwidth=2, relief='groove')
        self.show_panel_1.place(x=PADDINGS[1], y=TOOLBOX_AREA[1]+PADDINGS[2]+PADDINGS[1])

        # palette
        if self.RGB:
            self.palette_panel = Label(self.root, width=256, height=256, borderwidth=2, relief='groove')
            self.palette_panel.place(x=PADDINGS[1]+PADDINGS[2]*2+self.SHOW_AREA[0]+DRAWING_AREA[0], y=TOOLBOX_AREA[1]+PADDINGS[2]+PADDINGS[1])

            self.palette_square = Label(self.root, width=128, height=128, borderwidth=0, relief='groove')
            self.palette_square.place(x=PADDINGS[1]+PADDINGS[2]*2+self.SHOW_AREA[0]+DRAWING_AREA[0]+65, y=TOOLBOX_AREA[1]+PADDINGS[2]+PADDINGS[1]+65)

        ############################################################
        #                         Tool Box                         #
        ############################################################
        self.stencils = {'pen': 0, 'eraser':1, 'paint_bucket':2}

        self.minus_image = PhotoImage(file='resource/minus.png')
        self.minus_button = Button(self.root, image=self.minus_image, width=32, height=32, command=self.minus_button_click)
        self.minus_button.place(x=PADDINGS[1], y=PADDINGS[1])

        self.plus_image = PhotoImage(file='resource/plus.png')
        self.plus_button = Button(self.root, image=self.plus_image, width=32, height=32, command=self.plus_button_click)
        self.plus_button.place(x=PADDINGS[1]+PADDINGS[2]*2+64, y=PADDINGS[1])

        self.iter_label_text = StringVar()
        self.iter_label_text.set('3')
        self.iter_label = Label(self.root, textvariable=self.iter_label_text, font=('Arial', 20))
        self.iter_label.place(x=PADDINGS[1]+PADDINGS[2]+32+10, y=PADDINGS[1])

        self.pen_image = PhotoImage(file='resource/pen.png')
        self.pen_button = Button(self.root, width=32, height=32, image=self.pen_image, command=self.pen_button_cllick)
        self.pen_button.place(x=PADDINGS[1]+PADDINGS[2]+self.SHOW_AREA[0], y=PADDINGS[1])
        self.pen_button.configure(background='#CCCCCC')

        self.eraser_image = PhotoImage(file='resource/eraser.png')
        self.eraser_button = Button(self.root, image=self.eraser_image, width=32, height=32, command=self.eraser_button_click)
        self.eraser_button.place(x=PADDINGS[1]+PADDINGS[2]*2+self.SHOW_AREA[0]+32, y=PADDINGS[1])

        self.paint_bucket_button_image = PhotoImage(file='resource/PaintBucket.png')
        self.paint_bucket_button = Button(self.root, image=self.paint_bucket_button_image, width=32, height=32, command=self.paint_bucket_button_click)
        self.paint_bucket_button.place(x=PADDINGS[1]+PADDINGS[2]*3+self.SHOW_AREA[0]+32*2, y=PADDINGS[1])

        self.load_button_image = PhotoImage(file='resource/load.png')
        self.load_button = Button(self.root, image=self.load_button_image, width=32, height=32, command=self.load_button_click)
        self.load_button.place(x=PADDINGS[1]+PADDINGS[2]*4+self.SHOW_AREA[0]+32*3, y=PADDINGS[1])

        self.cat_button_image = PhotoImage(file='resource/cat.png')
        self.cat_button = Button(self.root, image=self.cat_button_image, width=32, height=32, command=self.cat_button_click)
        self.cat_button.place(x=PADDINGS[1]+PADDINGS[2]*5+self.SHOW_AREA[0]+32*4, y=PADDINGS[1])

        self.stencil_buttons = [self.pen_button, self.eraser_button, self.paint_bucket_button]
        ############################################################

        ###################### Drawing ######################
        self.img = np.ones((512, 512, 3), np.uint8) * 255
        self.img_re = np.ones(self.SHOW_AREA, np.uint8) * 255
        self.palette_img = np.ones((256, 256, 3)) * 240
        self.palette_square_image = np.zeros((128, 128, 3))
        self.palette_weighting_1 = np.zeros((128, 128, 3))
        self.palette_weighting_2 = np.zeros((128, 128, 3))
        self.stencil_size = np.ones(len(self.stencils), np.int) * 5
        self.stencil_id = 0
        self.mouse_x = self.mouse_y = self.pre_x = self.pre_y = 0
        self.palette_circle_x, self.palette_circle_y, self.palette_square_x, self.palette_square_y = 195, 33, 0, 127
        self.palette_pressing = False
        self.redo_stack = []

    ### Button events ###
    def minus_button_click(self):
        self.RECONSTRUCT_ITER -= 1
        self.RECONSTRUCT_ITER = max(self.RECONSTRUCT_ITER, 1)
        self.iter_label_text.set(str(self.RECONSTRUCT_ITER))

    def plus_button_click(self):
        self.RECONSTRUCT_ITER += 1
        # RECONSTRUCT_ITER = min(RECONSTRUCT_ITER, 10)
        self.iter_label_text.set(str(self.RECONSTRUCT_ITER))

    def highlight_button(self, id):
        for button in self.stencil_buttons:
            button.configure(background='white')
        self.stencil_buttons[id].configure(background='#CCCCCC')

    def pen_button_cllick(self):
        self.stencil_id = self.stencils['pen']
        self.highlight_button(self.stencil_id)

    def eraser_button_click(self):
        self.stencil_id = self.stencils['eraser']
        self.highlight_button(self.stencil_id)

    def paint_bucket_button_click(self):
        self.stencil_id = self.stencils['paint_bucket']
        self.highlight_button(self.stencil_id)

    def load_button_click(self):
        self.img = np.copy(np.array(Image.open('save/' + os.listdir('save/')[0]).resize((DRAWING_AREA[0], DRAWING_AREA[1]), Image.ANTIALIAS))[:,:,0:3])

    def cat_button_click(self):
        self.img = np.copy(np.array(Image.open('resource/big_cat.png'))[:,:,0:3])

    #########################################

    def insert_stack(self ,img):
        self.redo_stack.insert(0,np.copy(self.img))
        if(len(self.redo_stack) > 10):
            self.redo_stack.pop()

    def redo(self):
        if(len(self.redo_stack)>0):
            print("Redo")
            self.img = np.copy(self.redo_stack[0])
            self.redo_stack.pop(0)
        else:
            print("Can't Redo")

    def reImage(self ,img):
        if not self.RGB:
            return np.array(Image.fromarray(self.img).resize(self.INPUT_SIZE, Image.ANTIALIAS))[:,:,0] / 255.
        return np.array(Image.fromarray(self.img).resize(self.INPUT_SIZE, Image.ANTIALIAS))/ 255.

    def search_color(self, color, circle_color):
        for i in range(-120, 120):
            for j in range(-120, 120):
                if abs(i) <= 66 and abs(j) <= 66:
                    continue
                if np.linalg.norm((i, j)) < 116 and np.linalg.norm((i, j)) > 114 and np.linalg.norm(np.array(circle_color-self.palette_img[i+127][j+127])) < 2:
                    # print(palette_img[i+127][j+127])
                    self.palette_circle_x = int(j/np.linalg.norm((i, j))*115+127)
                    self.palette_circle_y = int(i/np.linalg.norm((i, j))*115+127)
                    update_palette_square()
                    for n in range(128):
                        for m in range(128):
                            if np.linalg.norm(np.array(self.palette_square_image[n][m]- color)) < 1:
                                self.palette_square_x = m
                                self.palette_square_y = n
                                self.update_palette()
                                self.update_palette_square()
                                return
                                

    def get_color(self, color):
        # print(color)
        tmp = color + 255 - max(color)
        # print(tmp)
        if min(tmp) == 255:
            self.search_color(color, (255, 255, 0))
            return
        # print('!')
        if tmp[0] == 255:
            if tmp[1] > tmp[2]:
                tmp[1] -= (255-tmp[1])*(tmp[2]/(255-tmp[2]))
                self.search_color(color, (tmp[0], tmp[1], 0))
            else:
                tmp[2] -= (255-tmp[2])*(tmp[1]/(255-tmp[1]))
                self.search_color(color, (tmp[0], 0, tmp[2]))
        elif tmp[1] == 255:
            if tmp[0] > tmp[2]:
                tmp[0] -= (255-tmp[0])*(tmp[2]/(255-tmp[2]))
                self.search_color(color, (tmp[0], tmp[1], 0))
            else:
                tmp[2] -= (255-tmp[2])*(tmp[0]/(255-tmp[0]))
                self.search_color(color, (0, tmp[1], tmp[2]))
        else:
            if tmp[0] > tmp[1]:
                tmp[0] -= (255-tmp[0])*(tmp[1]/(255-tmp[1]))
                self.search_color(color, (tmp[0], 0, tmp[2]))
            else:
                tmp[1] -= (255-tmp[1])*(tmp[0]/(255-tmp[0]))
                self.search_color(color, (0, tmp[1], tmp[2]))
        # print(tmp)

    def Filtering(self, img, k=5):
        blur = cv2.GaussianBlur(img,(k,k),0)
        return blur

    def fill_area(self, p, color):
        p = (p[1], p[0])
        m = np.zeros((img.shape[0], img.shape[1]))
        c = np.copy(img[p[0]][p[1]])
        s = set([])
        s.add(p)
        dir = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        cnt = 0
        while len(s) != 0:
            cnt += 1
            cur = s.pop()
            self.img[cur[0]][cur[1]] = np.copy(color)
            for d in dir:
                if cur[0]+d[0] >= 0 and cur[0]+d[0] < DRAWING_AREA[0] and cur[1]+d[1] >= 0 and cur[1]+d[1] < DRAWING_AREA[1] and m[cur[0]+d[0]][cur[1]+d[1]] == 0 and np.array_equal(img[cur[0]+d[0]][cur[1]+d[1]], c):
                    s.add((cur[0]+d[0], cur[1]+d[1]))
                    m[cur[0]+d[0]][cur[1]+d[1]] = 1
        print(cnt)

    def draw(self, p1, p2):
        if self.stencil_id == self.stencils['eraser']:
            cv2.line(self.img,p1,p2,(255, 255, 255),self.stencil_size[self.stencil_id])
        elif self.stencil_id == self.stencils['pen']:
            if self.RGB:
                cv2.line(self.img,p1,p2,tuple(self.palette_square_image[self.palette_square_y][self.palette_square_x]),self.stencil_size[self.stencil_id])
            else:
                cv2.line(self.img,p1,p2,(0, 0, 0),self.stencil_size[self.stencil_id])
        elif self.stencil_id == self.stencils['paint_bucket']:
            self.fill_area(p2, self.palette_square_image[self.palette_square_y][self.palette_square_x])

    def update_drawing_panel(self):
        img_out = np.copy(self.img)
        cv2.circle(img_out,(self.mouse_x,self.mouse_y),int(self.stencil_size[self.stencil_id]/2+1),(0,0,0),1)
        cv2.circle(img_out,(self.mouse_x,self.mouse_y),int(self.stencil_size[self.stencil_id]/2),(255,255,255),1)
        photo = ImageTk.PhotoImage(image=Image.fromarray(img_out))
        self.drawing_panel.configure(image=photo)
        self.drawing_panel.image = photo

    def update_showing_panel(self):
        photo = ImageTk.PhotoImage(image=Image.fromarray(self.img_re))
        self.show_panel_1.configure(image=photo)
        self.show_panel_1.image = photo

    def get_circle_color(self, x, y):
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

    def change_stencil_size(self, n):
        if self.stencil_id == self.stencils['paint_bucket']:
            return

        self.stencil_size[self.stencil_id] += n
        self.stencil_size[self.stencil_id] = max(2, self.stencil_size[self.stencil_id])

    def create_palette(self):
        for i in range(128):
            for j in range(128):
                self.palette_weighting_1[i][j] = (127-j)/127
                self.palette_weighting_2[i][j] = (i)/127*255

        for i in range(-127, 128):
            for j in range(-127, 128):
                if (np.linalg.norm((i, j)) > 125 and np.linalg.norm((i, j)) < 126.5) or (np.linalg.norm((i, j)) > 104 and np.linalg.norm((i, j)) < 105.5):
                    self.palette_img[i+127][j+127] = 255
                elif np.linalg.norm((i, j)) < 125 and np.linalg.norm((i, j)) > 105.5:
                    self.palette_img[i+127][j+127] = self.get_circle_color(i, j)
                elif (np.linalg.norm((i, j)) > 126.5 and np.linalg.norm((i, j)) < 128) or (np.linalg.norm((i, j)) > 102.5 and np.linalg.norm((i, j)) < 104):
                    self.palette_img[i+127][j+127] = 200

                if (i== -66 or i == 66) and (j >= -66 and j <= 66):
                    self.palette_img[i+127][j+127] = 150
                if (j== -66 or j == 66) and (i >= -66 and i <= 66):
                    self.palette_img[i+127][j+127] = 150

    def update_palette(self):
        img_out = np.copy(self.palette_img).astype(np.uint8)
        cv2.circle(img_out,(self.palette_circle_x,self.palette_circle_y),4,(0,0,0),2)
        cv2.circle(img_out,(self.palette_square_x+63,self.palette_square_y+65),4,(0,0,0),2)
        photo = ImageTk.PhotoImage(image=Image.fromarray(img_out))
        self.palette_panel.configure(image=photo)
        self.palette_panel.image = photo

    def update_palette_square(self):
        self.palette_square_image = (-self.palette_img[self.palette_circle_y][self.palette_circle_x]+255)*self.palette_weighting_1 + self.palette_img[self.palette_circle_y][self.palette_circle_x]
        self.palette_square_image[self.palette_square_image>255] = 255
        self.palette_square_image -= self.palette_weighting_2
        self.palette_square_image[self.palette_square_image<0] = 0

        img_out = np.copy(self.palette_square_image).astype(np.uint8)
        if max(self.palette_square_image[self.palette_square_y][self.palette_square_x]) > 128:
            cv2.circle(img_out,(self.palette_square_x,self.palette_square_y),4,(0,0,0),2)
        else:
            cv2.circle(img_out,(self.palette_square_x,self.palette_square_y),4,(256,256,256),2)
        photo = ImageTk.PhotoImage(image=Image.fromarray(img_out))
        self.palette_square.configure(image=photo)
        self.palette_square.image = photo

    def save_result(self, iter=5):
        print('Save...')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        img_out = self.reImage(self.img)
        for i in range(iter+1):
            if not self.RGB:
                plt.imsave(self.save_dir + str(i) + '.png',img_out.reshape(img_out.shape[0], img_out.shape[1]),cmap='Greys_r')
            else:
                plt.imsave(self.save_dir + str(i) + '.png',img_out)
            img_out = self.model.AutoDraw(img_out)

    def set_root_events(self):
        def key(event):
            if event.char == 's' or event.char == 'S':
                self.save_result()
            elif event.char == 'c' or event.char == 'C':
                self.insert_stack(self.img)
                self.img = np.ones((512, 512, 3), np.uint8) * 255
            elif event.char == 'p' or event.char == 'P':
                self.pen_button_cllick()
            elif event.char == 'e' or event.char == 'E':
                self.eraser_button_click()
            elif event.char == '1':
                self.change_stencil_size(-2)
            elif event.char == '2':
                self.change_stencil_size(2)
            elif event.char == 'r' or event.char == 'R':
                self.redo()

        self.root.bind('<Key>', key)

    def set_drawing_panel_events(self):
        def left_button_click(event):
            self.insert_stack(self.img)
            self.pre_x = event.x
            self.pre_y = event.y
            mouse_motion(event)
            if self.stencil_id == self.stencils['paint_bucket']:
                self.draw((self.pre_x, self.pre_y), (event.x, event.y))


        def right_button_click(event):
            self.get_color(self.img[event.y][event.x])

        def left_button_move(event):
            if self.stencil_id != self.stencils['paint_bucket']:
                self.draw((self.pre_x, self.pre_y), (event.x, event.y))   
            self.pre_x = event.x
            self.pre_y = event.y
            mouse_motion(event)

        def wheel(event):
            if event.delta > 0:
                self.change_stencil_size(2)
            else:
                self.change_stencil_size(-2)

        def mouse_motion(event):
            self.mouse_x, self.mouse_y = event.x, event.y

        self.drawing_panel.bind('<Button-1>', left_button_click)
        self.drawing_panel.bind('<Button-3>', right_button_click)
        self.drawing_panel.bind('<B1-Motion>', left_button_move)
        self.drawing_panel.bind('<MouseWheel>', wheel)
        self.drawing_panel.bind('<Motion>', mouse_motion)

    def set_palette_event(self):
        def update_palate_position(event):
            if np.linalg.norm((event.x-127, event.y-127)) < 125 and np.linalg.norm((event.x-127, event.y-127)) > 105.5:
                self.palette_pressing = True

            if not self.palette_pressing:
                return

            self.palette_circle_x, self.palette_circle_y = event.x, event.y
            t = np.linalg.norm((self.palette_circle_x-127, self.palette_circle_y-127))
            self.palette_circle_x = int((self.palette_circle_x-127) / t * 115) + 127
            self.palette_circle_y = int((self.palette_circle_y-127) / t * 115) + 127
            self.update_palette()
            self.update_palette_square()
                    

        def left_button_click(event):
            update_palate_position(event)

        def left_button_release(event):
            self.palette_pressing = False
            
        def left_button_move(event):
            update_palate_position(event)
        self.palette_panel.bind('<Button-1>', left_button_click)
        self.palette_panel.bind('<B1-Motion>', left_button_move)
        self.palette_panel.bind('<ButtonRelease-1>', left_button_release)

    def set_palette_square_event(self):
        def update_palate_position(event):
            self.palette_square_x, self.palette_square_y = min(max(event.x,0),127), min(max(event.y,0),127)
            self.update_palette_square()
            self.update_palette()
                    
        def left_button_click(event):
            update_palate_position(event)

            
        def left_button_move(event):
            update_palate_position(event)

        self.palette_square.bind('<Button-1>', left_button_click)
        self.palette_square.bind('<B1-Motion>', left_button_move)

    def main(self):
        self.create_palette()
        self.update_palette()
        self.update_palette_square()
        self.set_root_events()
        self.set_palette_event()
        self.set_drawing_panel_events()
        self.set_palette_square_event()
        cnt = 0
        while True:
            if cnt > self.update_freq:
                if self.USE_MODEL:
                    self.img_re = self.reImage(self.img)
                    self.img_re = self.model.AutoDraw(self.img_re, iter=self.RECONSTRUCT_ITER)*255
                    if self.RGB:
                        self.img_re = np.reshape(self.img_re, (self.INPUT_SIZE[0],self.INPUT_SIZE[1],3)).astype(np.uint8)
                        self.img_re = np.array(Image.fromarray(self.img_re).resize((self.SHOW_AREA[0], self.SHOW_AREA[1]), Image.ANTIALIAS))
                    else:
                        self.img_re = np.reshape(self.img_re, self.INPUT_SIZE).astype(np.uint8)
                        self.img_re = np.array(Image.fromarray(self.img_re, 'L').resize((self.SHOW_AREA[0], self.SHOW_AREA[1]), Image.ANTIALIAS))
                cnt = 0
            cnt += 1

            self.update_drawing_panel()
            self.update_showing_panel()
            self.root.update()
        self.root.mainloop()