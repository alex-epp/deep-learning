import tkinter
import numpy as np


class DrawingCanvas:
    def __init__(self, width, height):
        self.is_mouse_pressed = False
        self.mouse_x = 0
        self.mouse_y = 0
        self.width = width
        self.height = height

        self.root = tkinter.Tk()
        self.canvas = tkinter.Canvas(
            self.root,
            background='black',
            width=self.width,
            height=self.height,
        )
        self.canvas.pack()
        self.canvas.bind('<Motion>', self.mouse_motion)
        self.canvas.bind('<ButtonPress-1>', self.mouse_down)
        self.canvas.bind('<ButtonRelease-1>', self.mouse_up)

        self.clear_button = tkinter.Button(
            self.root,
            text='Clear',
            command=self.clear,
        )
        self.clear_button.pack()

        self.send_button = tkinter.Button(
            self.root,
            text='Send',
            command=self.send,
        )
        self.send_button.pack()

    def loop(self):
        self.root.mainloop()

    def mouse_motion(self, event):
        if self.is_mouse_pressed:
            event.widget.create_line(
                self.mouse_x, self.mouse_y,
                event.x, event.y,
                smooth=True,
                fill='white',
            )

        self.mouse_x = event.x
        self.mouse_y = event.y

    def mouse_down(self, event):
        self.is_mouse_pressed = True

    def mouse_up(self, event):
        self.is_mouse_pressed = False

    def clear(self):
        self.canvas.delete('all')

    def send(self):
        self.canvas.postscript(
            file='tmp/img.ps',
            colormode='greyscale'
        )


dc = DrawingCanvas(200, 200)
dc.loop()
