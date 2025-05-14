import tkinter
try:
    from cv2 import cv2
except ImportError:
    import cv2
from PIL import Image, ImageTk


class ScrollableImage(tkinter.Frame):
    def __init__(self, master, *args, **kwargs):
        self.width = kwargs.pop("width", 400)
        self.height = kwargs.pop("height", 400)
        self.command = kwargs.pop("command", lambda: None)
        self.master = master
        super().__init__(master, *args, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.canvas = tkinter.Canvas(self, bg="#ffffff", height=self.height, width=self.width)
        self.canvas.grid(row=0, column=0, sticky="news")
        self.hbar = tkinter.Scrollbar(self, orient=tkinter.HORIZONTAL)
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.hbar.config(command=self.xview)
        self.vbar = tkinter.Scrollbar(self, orient=tkinter.VERTICAL)
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.vbar.config(command=self.yview)
        self.canvas.config(scrollregion=(0, 0, 500, 500))
        self.canvas.bind("<MouseWheel>", self.on_scroll)
        self.canvas.bind("<Button-4>", self.on_scroll)
        self.canvas.bind("<Button-5>", self.on_scroll)

        self.fac = 1
        self.scale = 1
        self.xslice = slice(0, 0)
        self.yslice = slice(0, 0)
        self.img_origin = [0, 0]
        self.target_size = (0, 0)
        self.image_id = None
        self.image = None
        self.photoimage = None
        self.iw = 0
        self.ih = 0

    def update_image(self):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw==0 or ch==0 or self.iw==0 or self.ih==0: return
        self.fac = min(cw/self.iw, ch/self.ih)
        vw = min(self.iw, int(cw/(self.scale*self.fac)))
        vh = min(self.ih, int(ch/(self.scale*self.fac)))
        if self.img_origin[0]+vw >= self.iw: self.img_origin[0] = self.iw - vw
        if self.img_origin[0] < 0: self.img_origin[0] = 0
        if self.img_origin[1]+vh >= self.ih: self.img_origin[1] = self.ih - vh
        if self.img_origin[1] < 0: self.img_origin[1] = 0
        ox, oy = self.img_origin
        self.xslice = slice(ox, ox+vw)
        self.yslice = slice(oy, oy+vh)
        self.target_size = (
            min(cw, int(self.iw*self.fac*self.scale)),
            min(ch, int(self.ih*self.fac*self.scale))
        )
        self.vbar.set(self.yslice.start/self.ih, self.yslice.stop/self.ih)
        self.hbar.set(self.xslice.start/self.iw, self.xslice.stop/self.iw)
        # print("target {} cw {} ch {}".format(self.target_size, cw, ch))
        self.command()
        if self.image is not None:
            #print("writing image with size", self.image.shape)
            img_s = Image.fromarray(self.image[:, :, ::-1])
            self.photoimage = ImageTk.PhotoImage(image=img_s)
            if self.image_id is not None:
                self.canvas.itemconfigure(self.image_id, image=self.photoimage)
            else:
                self.image_id = self.canvas.create_image(0, 0, image=self.photoimage, anchor="nw")

    def yview(self, *args):
        ofs = float(args[1])
        self.img_origin[1] = int(ofs*self.ih)
        self.update_image()

    def xview(self, *args):
        ofs = float(args[1])
        self.img_origin[0] = int(ofs*self.iw)
        self.update_image()

    def transform(self, img):
        img = img[self.yslice, self.xslice, :]
        return cv2.resize(img, self.target_size)

    def set_image(self, img):
        self.image = img

    def set_size(self, iw, ih):
        self.iw = iw
        self.ih = ih
        # print("Image size is", iw, ih)
        self.update_image()

    def canvas_to_image(self, x, y):
        return [
            x/(self.scale*self.fac) + self.img_origin[0],
            y/(self.scale*self.fac) + self.img_origin[1]
        ]

    def image_to_canvas(self, x, y):
        return [
            (x - self.img_origin[0]) * self.scale * self.fac,
            (y - self.img_origin[1]) * self.scale * self.fac
        ]

    def on_scroll(self, evt):
        if evt.num == 5:
            evt.delta = -1
        if evt.num == 4:
            evt.delta = 1
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        d = (-evt.delta, 0) if evt.state & 1 else (0, -evt.delta)
        zoom = (evt.state & 0b1100) > 0
        x = evt.x_root - self.canvas.winfo_rootx()
        y = evt.y_root - self.canvas.winfo_rooty()
        # Find the image coordinates under the cursor
        # scroll_origin is the number of image pixels between the screen origin and the cursor
        scroll_origin = self.canvas_to_image(x, y)
        if zoom and (evt.state & 0b0001) == 0:
            if d[1] > 0:
                self.scale /= 1.125
            else:
                self.scale *= 1.125
            # Solve for the origin that would align the scroll_image location to the cursor
            # evt.x/(self.scale*self.fac) + new_origin_x = scroll_origin[0]
            # scroll_origin[0] - evt.x/(self.scale*self.fac) = new_origin_x
            self.img_origin[0] = int(scroll_origin[0] - x/(self.scale*self.fac))
            self.img_origin[1] = int(scroll_origin[1] - y/(self.scale*self.fac))
            if self.scale < 1: self.scale = 1
            self.update_image()
        elif not zoom:
            if evt.state & 0b0001: self.img_origin[0] += int(-evt.delta*100/self.scale)
            else: self.img_origin[1] += int(-evt.delta*100/self.scale)
            self.update_image()

    def bind(self, *args, **kwargs):
        self.canvas.bind(*args, **kwargs)


if __name__ == '__main__':
    orig = cv2.imread("input.jpg")
    tk = tkinter.Tk()
    s = ScrollableImage(tk, width=400, height=400, command=lambda: s.set_image(s.transform(orig)))
    tk.rowconfigure(0, weight=1)
    tk.columnconfigure(0, weight=1)
    s.grid(row=0, column=0, sticky="news")
    tk.update()
    s.set_size(*orig.shape[1::-1])
    s.set_image(s.transform(orig))
    tk.bind("<Configure>", lambda x: s.update_image())
    tk.mainloop()
