import MLP
import MLP.mnist

import tkinter
import numpy as np
import PIL
import PIL.ImageOps
import io


class DrawingCanvas:
    def __init__(self, width, height, notify):
        self.is_mouse_pressed = False
        self.mouse_x = 0
        self.mouse_y = 0
        self.width = width
        self.height = height

        self.notify = notify

        self.root = tkinter.Tk()

        self.message_text = tkinter.StringVar(self.root, 'Sketch a digit')
        tkinter.Label(
            self.root,
            textvariable=self.message_text,
        ).pack()

        self.canvas = tkinter.Canvas(
            self.root,
            background='white',
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
            text='Read',
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
                fill='black',
                width=5,
            )

        self.mouse_x = event.x
        self.mouse_y = event.y

    def mouse_down(self, event):
        self.is_mouse_pressed = True

    def mouse_up(self, event):
        self.is_mouse_pressed = False

    def clear(self):
        self.canvas.delete('all')
        self.message_text.set('Sketch a digit')

    def send(self):
        ps = self.canvas.postscript(colormode='color')
        img = PIL.Image.open(io.BytesIO(ps.encode('utf-8')))
        img = PIL.ImageOps.invert(img)
        self.show_message(self.notify(img))

    def show_message(self, msg_string):
        self.message_text.set(msg_string)


def predict(mlp, img):
    ex = MLP.mnist.from_image(img)
    # MLP.mnist.visualize(ex)
    probabilities = mlp.eval(ex.T)
    prediction = np.argmax(probabilities)
    probability = probabilities[prediction, 0]*100
    return '{} ({:.2f}%)'.format(prediction, probability)


filenames = ['saves/mlp{}'.format(i+1) for i in range(5)]
ensemble = MLP.Ensemble()
ensemble.load(open(n, 'rb') for n in filenames)

dc = DrawingCanvas(200, 200, lambda image: predict(ensemble, image))
dc.loop()
