import json
import tkinter
import typing
import zipfile
import os.path

import numpy
try:
    from cv2 import cv2
except ImportError:
    import cv2
from PIL import ImageTk, Image
import scrollable_image
import base64


class ProjectSetupDialog(tkinter.Toplevel):
    def __init__(self, solder_fname, comp_fname, project_name, on_complete, save=True):
        """
        Dialog for distorting one image to line up with another

        :param solder_fname: Filename or array of solder side image
        :param comp_fname: Filename or array of component side image
        :param project_name: Filename for saving project file. Can be "" if save=False
        :param on_complete: Function that is called when the user is done
        :param save: Determines whether the final images are stored to a project file when done.
        """
        super().__init__()
        print("Initializing setup dialog")
        self.photo = []
        self.on_complete = on_complete
        self.save = save
        self.solder_marks = [] # [(495, 235), (2397, 231), (1812, 3088), (534, 3304)]
        self.comp_marks = [] # [(495, 235), (2390, 231), (1809, 3115), (521, 3329)]
        self.comp_idx = []
        self.solder_idx = []
        self.solder_active = []
        self.comp_active = []
        self.show_comp = False
        self.solder_arr = (cv2.imread(solder_fname) if isinstance(solder_fname, str) else solder_fname)[:, :, ::-1]
        self.comp_arr = (cv2.imread(comp_fname) if isinstance(comp_fname, str) else comp_fname)[:, :, ::-1]
        print("loaded images")
        self.solder_fname = solder_fname
        self.comp_fname = comp_fname
        self.project_name = project_name
        self.solder_photo = None
        self.comp_photo = None
        self.solder_id = None
        self.comp_id = None
        self.fac_s = 1
        self.fac_c = 1
        self.canvas_lastsize = []
        print("configuring")
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        print("Setting up {} and {}")

        tkinter.Label(self, text="""Select 4 points on the solder side image
Then select corresponding points on the component side""").grid(row=0, column=0, columnspan=3)

        self.canvas = scrollable_image.ScrollableImage(self, command=self.on_scroll)
        self.canvas.grid(row=1, column=0, columnspan=3, sticky="news")

        tkinter.Button(self, text="View other side", command=self.switch_sides).grid(row=2, column=0, sticky="e")
        tkinter.Button(self, text="Clear markers", command=self.clear_markers).grid(row=2, column=1)
        tkinter.Button(self, text="Continue", command=self.process_images).grid(row=2, column=2)

        self.update()
        self.canvas.set_size(*self.solder_arr.shape[1::-1])
        self.canvas.canvas.bind("<Configure>", lambda x: self.canvas.update_image())
        self.canvas.bind("<Motion>", self.move_marker)
        self.canvas.bind("<Button-1>", self.add_marker)
        self.solder_active = self.draw_marker(0, 0, 1)
        self.comp_active = self.draw_marker(0, 0, 1, state="hidden")

    def on_scroll(self):
        img = (self.comp_arr if self.show_comp else self.solder_arr)
        print("image is", img.shape)
        self.canvas.set_image(self.canvas.transform(img[:, :, ::-1]))
        for i, p, m in zip(range(len(self.comp_marks)), self.comp_marks, self.comp_idx):
            x, y = self.canvas.image_to_canvas(*p)
            self.move_marker(x=int(x), y=int(y), i=m, n=i+1)
        for i, p, m in zip(range(len(self.solder_marks)), self.solder_marks, self.solder_idx):
            x, y = self.canvas.image_to_canvas(*p)
            self.move_marker(x=int(x), y=int(y), i=m, n=i+1)

    def draw_marker(self, x, y, n, **kwargs):
        return [
            self.canvas.canvas.create_oval(x-3, y-3, x+3, y+3, **kwargs),
            self.canvas.canvas.create_line(x, (y-11 if n >= 1 else y-8), x, y-3, fill="purple", width=3, **kwargs),
            self.canvas.canvas.create_line(x+3, y, (x+11 if n >= 2 else x+8), y, fill="purple", width=3, **kwargs),
            self.canvas.canvas.create_line(x, y+3, x, (y+11 if n >= 3 else y+8), fill="purple", width=3, **kwargs),
            self.canvas.canvas.create_line((x-11 if n >= 4 else x-8), y, x-3, y, fill="purple", width=3, **kwargs)
        ]

    def add_marker(self, event):
        print("Clicked at {},{}. fac_s {}, fac_c {}".format(event.x, event.y, self.fac_s, self.fac_c))
        print("Component size {}".format(self.comp_arr.shape))
        print("Solder size {}".format(self.solder_arr.shape))
        ex, ey = self.canvas.canvas_to_image(event.x, event.y)
        if self.show_comp:
            self.comp_marks.append((int(ex), int(ey)))
            n = len(self.comp_marks)+1
            self.comp_idx.append(self.comp_active)
            if n <= 4: self.comp_active = self.draw_marker(event.x, event.y, n)
            else: self.comp_active = []
        else:
            self.solder_marks.append((int(ex), int(ey)))
            n = len(self.solder_marks)+1
            self.solder_idx.append(self.solder_active)
            if n <= 4: self.solder_active = self.draw_marker(event.x, event.y, n)
            else: self.solder_active = []

    def move_marker(self, event=None, x=None, y=None, i=None, n=None):
        x, y = x if x is not None else event.x, y if y is not None else event.y
        n = n if n is not None else (len(self.comp_marks) if self.show_comp else len(self.solder_marks))+1
        new_coords = [
            (x - 3, y - 3, x + 3, y + 3),
            (x, (y-11 if n >= 1 else y-8), x, y-3),
            (x+3, y, (x+11 if n >= 2 else x+8), y),
            (x, y + 3, x, (y + 11 if n >= 3 else y + 8)),
            ((x-11 if n >= 4 else x-8), y, x-3, y),
        ]
        for idx, c in zip(i or (self.comp_active if self.show_comp else self.solder_active), new_coords):
            self.canvas.canvas.coords(idx, *c)

    def switch_sides(self):
        print("solder", self.solder_idx, "comp", self.comp_idx)
        if self.show_comp:
            self.canvas.canvas.itemconfigure(self.solder_id, state="normal")
            self.canvas.set_size(*self.solder_arr.shape[1::-1])
            self.canvas.canvas.itemconfigure(self.comp_id, state="hidden")
            self.show_comp = False
            for x in sum(self.comp_idx, [])+self.comp_active: self.canvas.canvas.itemconfigure(x, state="hidden")
            for x in sum(self.solder_idx, [])+self.solder_active: self.canvas.canvas.itemconfigure(x, state="normal")
        else:
            self.canvas.canvas.itemconfigure(self.solder_id, state="hidden")
            self.canvas.canvas.itemconfigure(self.comp_id, state="normal")
            self.canvas.set_size(*self.comp_arr.shape[1::-1])
            self.show_comp = True
            for x in sum(self.comp_idx, [])+self.comp_active: self.canvas.canvas.itemconfigure(x, state="normal")
            for x in sum(self.solder_idx, [])+self.solder_active: self.canvas.canvas.itemconfigure(x, state="hidden")
        self.canvas.update_image()

    def clear_markers(self):
        if self.show_comp:
            for x in self.comp_idx: self.canvas.canvas.delete(x)
            self.comp_idx = []
            self.comp_marks = []
        else:
            for x in self.solder_idx: self.canvas.canvas.delete(x)
            self.solder_idx = []
            self.solder_marks = []

    def process_images(self, command=None):
        print("Component side marks", self.comp_marks)
        print("Solder side marks", self.solder_marks)
        if len(self.comp_marks) == len(self.solder_marks) == 4:
            tr = cv2.getPerspectiveTransform(numpy.float32(self.comp_marks), numpy.float32(self.solder_marks))
            result = cv2.warpPerspective(self.comp_arr[:, :, ::-1], tr, dsize=self.solder_arr.shape[1::-1])
            print(result.shape)
            if self.save:
                cv2.imwrite("/tmp/corrected.png", result)
                cv2.imwrite("/tmp/solder.png", self.solder_arr[:, :, ::-1])
                cv2.imwrite("/tmp/LAYERBottom.png", numpy.zeros(self.solder_arr.shape))
                with zipfile.ZipFile(self.project_name, "w") as z:
                    z.write("/tmp/solder.png", "solder.png")
                    z.write("/tmp/corrected.png", "comp.png")
                    z.write("/tmp/LAYERBottom.png", "LAYERBottom.png")
                    z.writestr("/tmp/setup.json", "")
                self.on_complete()
            else:
                self.on_complete(self.solder_arr[:, :, ::-1], result)


if __name__ == '__main__':
    tk = tkinter.Tk()
    setup = ProjectSetupDialog("/tmp/solder.jpeg",
                               "/tmp/component.jpeg", "/tmp/test.zip", print)
    tk.mainloop()
