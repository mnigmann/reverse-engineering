import tkinter
import zipfile
from tkinter import simpledialog

import numpy
import scipy.interpolate
import scipy.optimize
import scipy.spatial
try:
    from cv2 import cv2
except ImportError:
    import cv2

import layer
import scrollable_image


class ImageProcessorDialog(tkinter.Toplevel):
    def __init__(self, boundary, layers, img, command):
        super().__init__()
        self.layers = layers
        self.boundary = boundary
        self._img = img
        self.img = self._img[0]
        self.disp_img = self.img.copy()
        self.command = command
        self.rowconfigure(2, weight=1)
        self.columnconfigure(2, weight=1)

        tkinter.Label(self, text="Source image").grid(row=0, column=0, columnspan=2)
        self.source = tkinter.StringVar()
        self.source.set("solder")
        self.source.trace("w", lambda *args: self.set_blur(None))
        self.source_menu = tkinter.OptionMenu(self, self.source, "solder", "component")
        self.source_menu.grid(row=0, column=2)


        tkinter.Label(self, text="Gaussian blur radius:").grid(row=1, column=0, columnspan=2)
        self.blur_radius = tkinter.IntVar()
        self.blur_radius.set(0)
        self.blur_radius_e = tkinter.Entry(self, textvariable=self.blur_radius)
        self.blur_radius_e.grid(row=1, column=2, sticky="w")
        self.blur_radius_e.bind("<KeyRelease>", self.set_blur)


        self.canvas = scrollable_image.ScrollableImage(self, command=self.on_scroll)
        self.canvas.grid(row=2, column=0, columnspan=3, sticky="news")

        self.method_options = tkinter.LabelFrame(self, text="Options")
        self.method_options.grid(row=3, column=0, columnspan=3, sticky="news")

        #tkinter.Label(self, text="Add to layer").grid(row=3, column=0, sticky="w")

        self.add_mode = tkinter.Button(self, text="Add to layer",
                                    command=lambda: [self.add_mode.config(text=(
                                        "Erase from layer" if self.add_mode.cget(
                                            "text") == "Add to layer" else "Add to layer")), self.update_idletasks()])
        self.add_mode.grid(row=4, column=0)
        self.layer_var = tkinter.StringVar()
        self.layers_menu = tkinter.OptionMenu(self, self.layer_var, *[x.name for x in self.layers if x.name != "__boundary__"])
        self.layers_menu.grid(row=4, column=1, sticky="w")

        tkinter.Button(self, text="Continue", command=self.finished).grid(row=4, column=2, sticky="e")

        self.update()
        self.canvas.set_size(*self.img.shape[1::-1])
        self.canvas.canvas.bind("<Configure>", lambda x: self.canvas.update_image())
        self.canvas.bind("<Motion>", self.motion)

    def set_blur(self, event):
        r = 2*self.blur_radius.get() + 1
        self.img = cv2.GaussianBlur(self._img[0 if self.source.get() == "solder" else 1], (r, r), 0)
        self.update_image()

    def on_scroll(self):
        self.canvas.set_image(self.canvas.transform(self.disp_img))

    def update_image(self):
        self.canvas.update_image()

    def motion(self, event):
        pass

    def finished(self):
        pass


class ThresholdDialog(ImageProcessorDialog):
    def __init__(self, boundary, layers, img, command):
        super().__init__(boundary, layers, img, command)
        tkinter.Label(self.method_options, text="Source:").grid(row=0, column=0)
        self.prop = tkinter.StringVar()
        self.prop_menu = tkinter.OptionMenu(self.method_options, self.prop, "Red", "Green", "Blue", "Hue", "Saturation", "Value", command=lambda x: self.update_image())
        self.prop.set("Red")
        self.prop_menu.grid(row=0, column=1, sticky="w")

        self.var = tkinter.IntVar()
        self.l1 = tkinter.Label(self.method_options, text="Threshold")
        self.l1.grid(row=1, column=0)
        self.pol_b = tkinter.Button(self.method_options, text="<",
                                    command=lambda: self.pol_b.config(text=(
                                        ">=" if self.pol_b.cget(
                                            "text") == "<" else "<")))
        self.pol_b.grid(row=1, column=1)
        self.thr_e = tkinter.Entry(self.method_options, textvariable=self.var)
        self.thr_e.grid(row=1, column=2)
        self.thr_e.bind("<KeyRelease>", lambda x: self.update_image())
        self.thr_s = tkinter.Scale(self.method_options, orient=tkinter.HORIZONTAL,
                                   showvalue=0, variable=self.var, from_=0, to=255,
                                   command=lambda x: self.update_image())
        self.thr_s.grid(row=1, column=3)

        self.l2 = tkinter.Label(self.method_options, text="x: 0, y: 0, red: 0")
        self.l2.grid(row=2, column=0, columnspan=4)

    def motion(self, event):
        ex, ey = self.canvas.canvas_to_image(event.x_root - self.canvas.winfo_rootx(),
                                             event.y_root - self.canvas.winfo_rooty())
        if ex >= self.disp_img.shape[1] or ey >= self.disp_img.shape[0]: return
        ex = int(ex)
        ey = int(ey)
        self.l2.config(text="x: {}, y: {}, {}: {}".format(ex, ey, self.prop.get().lower(), self.disp_img[ey, ex, 0]))
        self.update_idletasks()

    def update_image(self):
        print("updating")
        self.disp_img = numpy.zeros(self.img.shape, dtype=numpy.uint8)
        p = self.prop.get()
        if p in ["Hue", "Saturation", "Value"]:
            hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            if p == "Hue": self.disp_img[:, :, 0] = hsv[:, :, 0]
            elif p == "Saturation": self.disp_img[:, :, 0] = hsv[:, :, 1]
            elif p == "Value": self.disp_img[:, :, 0] = hsv[:, :, 2]
        else:
            if p == "Red": self.disp_img[:, :, 0] = self.img[:, :, 2]
            elif p == "Green": self.disp_img[:, :, 0] = self.img[:, :, 1]
            elif p == "Blue": self.disp_img[:, :, 0] = self.img[:, :, 0]

        if self.pol_b.cget("text") == "<":
            mask = self.disp_img[:, :, 0] < self.var.get()
        else:
            mask = self.disp_img[:, :, 0] >= self.var.get()
        mask = numpy.array(mask, dtype=numpy.uint8) * self.boundary.layer[:, :, 0]
        self.disp_img[:, :, 2] = numpy.array(mask * 255, dtype=numpy.uint8)
        self.canvas.update_image()

    def finished(self):
        l = [x for x in self.layers if x.name == self.layer_var.get()]
        if not l: return
        l = l[0]
        b = self.boundary.layer[:, :, 0]
        print("Adding to layer", l.name, l.color, b.shape, self.disp_img[:, :, 0].shape)
        v = [1, 1, 1] if self.add_mode.cget("text") == "Add to layer" else [0, 0, 0]
        if self.pol_b.cget("text") == "<":
            l.layer[numpy.array((self.disp_img[:, :, 0] < (self.var.get()*b)), dtype=numpy.bool)] = v
        else:
            l.layer[numpy.array(((self.disp_img[:, :, 0]*b) >= self.var.get()), dtype=numpy.bool)] = v
        print("Layer write done")
        # self.destroy()
        self.command()


class ColorDistanceThresholdDialog(ImageProcessorDialog):
    def __init__(self, boundary, layers, img, command):
        super().__init__(boundary, layers, img, command)
        self.var = tkinter.IntVar()
        self.target_color = []
        self.l1 = tkinter.Label(self.method_options, text="Radius")
        self.l1.grid(row=0, column=0)
        self.thr_e = tkinter.Entry(self.method_options, textvariable=self.var)
        self.thr_e.grid(row=0, column=1)
        self.thr_e.bind("<KeyRelease>", lambda x: self.update_image())
        self.thr_s = tkinter.Scale(self.method_options, orient=tkinter.HORIZONTAL,
                                   showvalue=0, variable=self.var, from_=0, to=255,
                                   command=lambda x: self.update_image())
        self.thr_s.grid(row=0, column=2)
        self.l3 = tkinter.Label(self.method_options, text="Target: R: 0, G: 0, B: 0")
        self.l3.grid(row=1, column=0, columnspan=3)
        self.l2 = tkinter.Label(self.method_options, text="x: 0, y: 0, R: 0, G: 0, B: 0")
        self.l2.grid(row=2, column=0, columnspan=3)

        self.canvas.canvas.bind("<Button-1>", self.clicked)

    def clicked(self, event):
        ex, ey = self.canvas.canvas_to_image(event.x_root - self.canvas.winfo_rootx(),
                                             event.y_root - self.canvas.winfo_rooty())
        if ex >= self.disp_img.shape[1] or ey >= self.disp_img.shape[0]: return
        self.target_color = list(self.img[int(ey)][int(ex)])
        self.l3.config(text="R: {}, G: {}, B: {}".format(*self.target_color[::-1]))

    def motion(self, event):
        ex, ey = self.canvas.canvas_to_image(event.x_root - self.canvas.winfo_rootx(),
                                             event.y_root - self.canvas.winfo_rooty())
        if ex >= self.disp_img.shape[1] or ey >= self.disp_img.shape[0]: return
        ex = int(ex)
        ey = int(ey)
        self.l2.config(text="x: {}, y: {}, R: {}, G: {}, B: {}".format(ex, ey, *self.img[ey, ex, ::-1]))
        self.update_idletasks()

    def update_image(self):
        self.disp_img = self.img
        img = numpy.array(self.img, dtype=numpy.int32)
        dist = (
                       (img[:, :, 0] - self.target_color[0])**2 +
                       (img[:, :, 1] - self.target_color[1])**2 +
                       (img[:, :, 2] - self.target_color[2])**2
               )
        ndist = numpy.zeros(dist.shape + (3,))
        ndist[:, :, 0] = dist
        ndist[:, :, 1] = dist
        ndist[:, :, 2] = dist
        t = self.var.get()**2
        self.disp_img = numpy.where(ndist < t, numpy.ones(self.disp_img.shape, dtype=numpy.uint8)*255, self.disp_img)
        self.canvas.update_image()

    def finished(self):
        img = numpy.array(self.img, dtype=numpy.int32)
        dist = (
                       (img[:, :, 0] - self.target_color[0])**2 +
                       (img[:, :, 1] - self.target_color[1])**2 +
                       (img[:, :, 2] - self.target_color[2])**2
               )
        t = self.var.get()**2
        l = [x for x in self.layers if x.name == self.layer_var.get()]
        if not l: return
        l = l[0]
        b = self.boundary.layer[:, :, 0]

        if self.add_mode.cget("text") == "Add to layer": l.layer[dist < b*t] = [1, 1, 1]
        else: l.layer[dist < b*t] = [0, 0, 0]
        self.command()


class GradientThresholdDialog(ImageProcessorDialog):
    def __init__(self, boundary, layers, img, command):
        super().__init__(boundary, layers, img, command)
        self.points = []
        self.last_gradient = None

        self.var = tkinter.IntVar()
        self.l1 = tkinter.Label(self.method_options, text="Threshold")
        self.l1.grid(row=1, column=0)
        self.pol_b = tkinter.Button(self.method_options, text="<",
                                    command=lambda: self.pol_b.config(text=(
                                        ">=" if self.pol_b.cget(
                                            "text") == "<" else "<")))
        self.pol_b.grid(row=1, column=1)
        self.thr_e = tkinter.Entry(self.method_options, textvariable=self.var)
        self.thr_e.grid(row=1, column=2)
        self.thr_e.bind("<KeyRelease>", lambda x: self.update_image())
        self.thr_s = tkinter.Scale(self.method_options, orient=tkinter.HORIZONTAL,
                                   showvalue=0, variable=self.var, from_=0, to=255,
                                   command=lambda x: self.update_image())
        self.thr_s.grid(row=1, column=3)

        tkinter.Button(self.method_options, text="Clear points", command=self.clear_points).grid(row=0, column=0, columnspan=2)

        tkinter.Label(self.method_options, text="Averaging radius").grid(row=2, column=0, columnspan=2)
        self.avg = tkinter.IntVar()
        self.avg_e = tkinter.Entry(self.method_options, textvariable=self.avg)
        self.avg_e.grid(row=2, column=2)
        self.avg.set(0)

        self.canvas.canvas.bind("<Button-1>", self.clicked)

    def update_image(self):
        self.disp_img = numpy.array(self.img)
        if len(self.points) < 4:
            self.canvas.update_image()
            return
        if self.last_gradient is not None:
            Z = self.last_gradient
        else:
            print("points:", self.points)
            sol = []
            rad = self.avg.get()
            for r, c in self.points:
                print(self.img[r, c], numpy.average(self.img[r-rad:r+rad+1, c-rad:c+rad+1], axis=(0,1)))
                sol.append(self.img[r, c])
            sol = numpy.array(sol)
            x_val = numpy.array([x[::-1] for x in self.points])
            print("Red points:", list(zip(x_val[:, 0], x_val[:, 1], sol[:, 2])))
            print("Green points:", list(zip(x_val[:, 0], x_val[:, 1], sol[:, 1])))
            print("Blue points:", list(zip(x_val[:, 0], x_val[:, 1], sol[:, 0])))
            print("Red   ", end=""); Z_r = self.fit_inverse(x_val, sol[:, 2])
            print("Green ", end=""); Z_g = self.fit_inverse(x_val, sol[:, 1])
            print("Blue  ", end=""); Z_b = self.fit_inverse(x_val, sol[:, 0])
            Z = cv2.merge((Z_b, Z_g, Z_r))
            self.last_gradient = Z
            cv2.imshow("gradient", Z)
        dist = numpy.where(self.disp_img > Z, self.disp_img - Z, Z - self.disp_img)
        dsum = cv2.cvtColor(dist, cv2.COLOR_BGR2GRAY)
        self.disp_img[dsum < self.var.get()*self.boundary.layer[:, :, 0]] = [255, 0, 255]
        self.canvas.update_image()

        solder = cv2.hconcat([self.disp_img, Z, dist])
        #cv2.imshow("Gradient reduction - inverse square", solder)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


    def eval_func(self, X, a, b, x_0, y_0):
        x = X[:, 0]
        y = X[:, 1]
        return b/((x-x_0)**2 + (y-y_0)**2 + a)**1.5

    def func_SSE(self, x, y, param):
        return numpy.sum((y - self.eval_func(x, *param))**2)

    def fit_inverse(self, x, y):
        try:
            result = scipy.optimize.differential_evolution(lambda p: self.func_SSE(x, y, p),
                                                           [[100000, 1000000], [100000, 1000000], [-1000, 1000],
                                                            [-1000, 1000]], seed=3).x
            fitted, pcov = scipy.optimize.curve_fit(self.eval_func, x, y, result)
            print(fitted, self.func_SSE(x, y, fitted))
            a, b, x_0, y_0 = fitted

            X = numpy.arange(0, self.img.shape[1])
            Y = numpy.arange(0, self.img.shape[0])
            X, Y = numpy.meshgrid(X, Y)
            Z = b / ((X - x_0) ** 2 + (Y - y_0) ** 2 + a) ** 1.5
            return Z.astype(numpy.uint8)
        except:
            return numpy.zeros(self.img.shape[:2], dtype=numpy.uint8)

    def clicked(self, event):
        ex, ey = self.canvas.canvas_to_image(event.x_root - self.canvas.winfo_rootx(),
                                             event.y_root - self.canvas.winfo_rooty())
        if ex >= self.disp_img.shape[1] or ey >= self.disp_img.shape[0]: return
        ex, ey = int(ex), int(ey)
        self.points.append([ey, ex])
        print([ey, ex], self.points)
        self.last_gradient = None
        self.update_image()

    def clear_points(self):
        self.points = []
        self.last_gradient = None
        self.update_image()

    def finished(self):
        l = [x for x in self.layers if x.name == self.layer_var.get()]
        if not l: return
        l = l[0]
        b = self.boundary.layer[:, :, 0]
        v = [1, 1, 1] if self.add_mode.cget("text") == "Add to layer" else [0, 0, 0]
        dist = numpy.where(self.img > self.last_gradient, self.img - self.last_gradient, self.last_gradient - self.img)
        dsum = cv2.cvtColor(dist, cv2.COLOR_BGR2GRAY)
        print("Adding to layer", l.name, l.color, (self.var.get()*b).dtype, dsum.dtype, v, self.pol_b.cget("text"))

        if self.pol_b.cget("text") == "<":
            l.layer[numpy.array(dsum < (self.var.get()*b), dtype=bool)] = v
        else:
            l.layer[(dsum*b) >= self.var.get()] = v
        print("Layer write done")
        # self.destroy()
        self.command()


class DelaunayThresholdDialog(ImageProcessorDialog):
    def __init__(self, boundary, layers, img, command):
        super().__init__(boundary, layers, img, command)

        tkinter.Label(self.method_options, text="Mode").grid(row=0, column=0)
        self.mode_var = tkinter.StringVar()
        self.mode_sel = tkinter.OptionMenu(self.method_options, self.mode_var, "Threshold", "Normalized threshold",
                                           "Relative normalized threshold")
        self.mode_sel.grid(row=0, column=1, columnspan=2, sticky="w")
        self.mode_var.set("Threshold")

        self.var = tkinter.IntVar()
        self.l1 = tkinter.Label(self.method_options, text="Radius")
        self.l1.grid(row=1, column=0)
        self.thr_e = tkinter.Entry(self.method_options, textvariable=self.var)
        self.thr_e.grid(row=1, column=1)
        self.thr_e.bind("<KeyRelease>", lambda x: self.update_scale(True))
        self.thr_s = tkinter.Scale(self.method_options, orient=tkinter.HORIZONTAL,
                                   showvalue=0, variable=self.var, from_=0, to=255,
                                   command=lambda x: self.update_scale(True))
        self.thr_s.grid(row=1, column=2)

        self.canvas.canvas.bind("<Button-1>", self.clicked)
        self.bind_all("<Control-k>", self.show_pred)
        self.bind_all("<Control-p>", self.manual_points)
        #self.points = [[3767, 2183], [3429, 1932], [3251, 2282], [3069, 1707], [2792, 1888], [2017, 1283], [1768, 1076], [1527, 774], [1429, 930], [1049, 1110], [1049, 842], [927, 638], [821, 479], [621, 401], [936, 298], [535, 748], [3580, 2655], [3495, 2406], [2196, 1894], [1678, 1340]]
        self.points = []
        self.simplices = None
        self.pred = self.img
        self.dist = numpy.ones(self.img.shape[:2])*255
        self.mask = numpy.zeros_like(self.dist)
        self.points_added = False
        self.last_threshold = -1
        self.update_image()

    def manual_points(self, evt):
        pts = simpledialog.askstring(title="Points", prompt="Enter points")
        self.points = eval(pts)
        self.points_added = True
        self.update_image()

    def show_pred(self, evt):
        cv2.imshow("img", numpy.uint8(self.pred))
        rdist = numpy.uint8(numpy.clip(numpy.sqrt(self.dist), 0, 255))
        cv2.imshow("dist", rdist)
        l = [x for x in self.layers if x.name == self.layer_var.get()][0]
        cv2.imwrite("/tmp/negative.png", numpy.uint8(255*self.boundary.layer*numpy.isfinite(self.pred)*(1-l.layer))[:, :, 0])
        cv2.imwrite("/tmp/dist.png", rdist)
        cv2.imwrite("/tmp/interpolated.png", self.pred)
        cv2.waitKey()
        cv2.destroyAllWindows()
        print(self.pred[:10, :10])
        # Test region: [1400:2400, 1290:2990]
        # Train region: [550:1450, 1200:-1448]

        # [[2618, 767], [2238, 785], [2043, 854], [2301, 1200], [2065, 1290], [1690, 1153], [1568, 993], [1460, 668], [1564, 459], [1191, 656], [1056, 399], [1058, 137], [1288, 89]]
        # [[2592, 553], [2699, 632], [2863, 940], [3020, 970], [3067, 1190], [2966, 1321], [2350, 1297], [2657, 1136], [2579, 1196], [2533, 1170], [2559, 1051], [2129, 547], [1754, 540], [1528, 545], [1092, 549], [1159, 602], [1193, 688], [1350, 1187], [1467, 1018], [1402, 816], [1759, 1357], [2421, 588], [2505, 685]]
        # [[2592, 553], [2699, 632], [2863, 940], [3020, 970], [3067, 1190], [2966, 1321], [2350, 1297], [2657, 1136], [2579, 1196], [2533, 1170], [2559, 1051], [2129, 547], [1754, 540], [1528, 545], [1092, 549], [1159, 602], [1193, 688], [1350, 1187], [1467, 1018], [1402, 816], [1759, 1357], [2421, 588], [2505, 685], [1247, 1362], [2802, 1456], [1012, 1285], [1187, 1040], [1205, 1452]]
        # [[1257, 1389], [2991, 1387], [3022, 2476], [1588, 2418], [2096, 1838], [1859, 1564], [2358, 1552], [2625, 1649], [2612, 2022], [2849, 1850], [2232, 2310], [2595, 2189], [1607, 1912], [1816, 2210], [2359, 1918], [1211, 2272], [1235, 2387]]

        # Full (RNT at 100)
        # [[1257, 1389], [2991, 1387], [3022, 2476], [1588, 2418], [2096, 1838], [1859, 1564], [2358, 1552], [2625, 1649], [2612, 2022], [2849, 1850], [2232, 2310], [2595, 2189], [1607, 1912], [1816, 2210], [2359, 1918], [1211, 2272], [1235, 2387], [2912, 1046], [2669, 1036], [2236, 1237], [2553, 1278], [1761, 1272], [1690, 1162], [1993, 972], [1594, 1010], [1502, 1141], [1402, 930], [1628, 602], [2238, 732], [2607, 864], [2632, 764], [3034, 952], [2580, 572], [1257, 653], [1083, 565], [970, 1128], [1043, 1958], [967, 373], [966, 201], [1261, 226], [1571, 254], [1557, 93]]

        # Power supply, RNT at 100
        # [[2775, 677], [1979, 792], [1439, 1198], [1366, 675], [1258, 639], [1034, 779], [1051, 1068], [914, 1411], [802, 2044], [385, 1993], [323, 1110], [289, 131], [658, 510], [1315, 456], [2476, 323], [2569, 141], [3556, 255], [3456, 1226], [3282, 790], [3332, 485], [2637, 1019], [1930, 1507], [1337, 1569], [3586, 1647], [228, 2205], [1729, 156], [540, 1433]]
    def clicked(self, event):
        ex, ey = self.canvas.canvas_to_image(event.x_root - self.canvas.winfo_rootx(),
                                             event.y_root - self.canvas.winfo_rooty())
        if ex >= self.disp_img.shape[1] or ey >= self.disp_img.shape[0]: return
        self.points.append([int(ex), int(ey)])
        self.points_added = True
        self.update_image()

    def set_blur(self, event):
        super().set_blur(event)
        self.disp_img = self.img.copy()
        self.update_image()

    def update_scale(self, reload=False):
        thr = self.var.get()
        if thr == self.last_threshold: return
        self.last_threshold = thr
        if thr == 0: return
        mode = self.mode_var.get()
        if mode == "Normalized threshold":
            dist = self.dist / numpy.linalg.norm(self.pred, axis=2)**2
            t = thr / 255
        elif mode == "Relative normalized threshold":
            dist = numpy.uint8(numpy.clip(255 * self.dist / numpy.linalg.norm(self.pred, axis=2)**2, 0, 255))
            t = thr / 255 * cv2.medianBlur(dist, 99)
        else:
            dist = self.dist
            t = thr ** 2
        print(self.mode_var.get())
        self.mask = dist < t
        self.disp_img = numpy.where(numpy.tensordot(dist < t, [1, 1, 1], axes=0), 255, self.img)

        if reload: self.update_image()

    def update_image(self):
        #self.disp_img = self.img.copy()
        print(self.points_added, self.points)
        if len(self.points) > 3 and self.points_added:
            self.disp_img = self.img.copy()
            self.points_added = False
            pts = numpy.array(self.points)
            res = scipy.spatial.Delaunay(pts)
            print(self.img.shape)
            print(self.img[pts[:, 1], pts[:, 0]])
            inter = scipy.interpolate.LinearNDInterpolator(self.points, self.img[pts[:, 1], pts[:, 0]])
            self.pred = inter(*numpy.meshgrid(numpy.arange(self.img.shape[1]), numpy.arange(self.img.shape[0])))
            self.dist = (self.pred[:, :, 0] - self.img[:, :, 0]) ** 2 + \
                        (self.pred[:, :, 1] - self.img[:, :, 1]) ** 2 + \
                        (self.pred[:, :, 2] - self.img[:, :, 2]) ** 2
            self.update_scale()
            self.simplices = res.simplices
        for p in self.points:
            cv2.circle(self.disp_img, p, 10, (0, 0, 255), -1)
        if self.simplices is not None:
            pts = numpy.array(self.points)
            for x in self.simplices:
                cv2.polylines(self.disp_img, [pts[x].reshape((-1, 1, 2))], True, (0, 0, 255), 1)
        self.canvas.update_image()

    def finished(self):
        img = numpy.array(self.img, dtype=numpy.int32)
        l = [x for x in self.layers if x.name == self.layer_var.get()]
        if not l: return
        l = l[0]
        b = self.boundary.layer[:, :, 0].astype(bool)

        if self.add_mode.cget("text") == "Add to layer": l.layer[b*self.mask] = [1, 1, 1]
        else: l.layer[b*self.mask] = [0, 0, 0]
        self.command()

def done(l):
    print("writing to file", l.name)
    img = numpy.zeros(solder.shape)
    bound.mode |= 0b0100
    img = bound.apply(img)
    l.mode |= 0b0100
    l.color = [255, 0, 255]
    print(numpy.sum(l.layer))
    img = l.apply(img)
    with open("/tmp/test_layer.png", "wb") as f: f.write(l.dumps())


if __name__ == '__main__':
    tk = tkinter.Tk()
    with zipfile.ZipFile("/Users/matthias/Documents/reverse_engineering/radio_interface.zip") as z:
        solder = cv2.imdecode(numpy.frombuffer(z.read("solder.png"), numpy.uint8), cv2.IMREAD_COLOR)
        comp = cv2.imdecode(numpy.frombuffer(z.read("comp.png"), numpy.uint8), cv2.IMREAD_COLOR)
        bottom = cv2.imdecode(numpy.frombuffer(z.read("LAYERBottom.png"), numpy.uint8), cv2.IMREAD_COLOR)
        bound = layer.Layer("__boundary__", z.read("boundary_layer.png"))
    print(solder.shape)
    out = layer.Layer("test", shape=solder.shape)
    DelaunayThresholdDialog(bound, [out], [solder, comp], lambda: done(out))
    tk.mainloop()

# [[3221, 2406], [2743, 2257], [3589, 658], [1598, 705], [1951, 2304], [1622, 1272], [1360, 2141], [1027, 2227], [845, 1873], [968, 1179], [1072, 1840], [402, 1521], [410, 1854], [499, 2349], [395, 2746], [1239, 2774], [1993, 2797], [3187, 2816], [3713, 2832], [3709, 849], [3310, 571], [2962, 1285], [2764, 1213], [2225, 408], [1593, 425], [1047, 416], [845, 476], [414, 615], [225, 2505], [247, 1018], [1633, 1182], [2055, 1164], [3319, 410], [648, 1516]]
