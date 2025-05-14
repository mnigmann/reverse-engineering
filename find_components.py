try:
    from cv2 import cv2
except ImportError:
    import cv2
import numpy
import auto_trace
import tkinter
import zipfile
import layer
import view_comp
import matplotlib.pyplot as plt

numpy.set_printoptions(linewidth=1000, threshold=30000)


def sseconv(img, ker):
    if img.ndim == 3:
        return sseconv(img[:, :, 0], ker[:, :, 0]) + sseconv(img[:, :, 1], ker[:, :, 1]) + sseconv(img[:, :, 2], ker[:, :, 2])
    #img = numpy.uint16(img)
    ker = numpy.uint16(ker)
    ones = numpy.ones_like(ker, dtype=numpy.uint8)
    return cv2.filter2D(img*img, cv2.CV_32F, ones) + numpy.sum(ker*ker) - 2*cv2.filter2D(img, cv2.CV_32F, ker)


class FindComponentsDialog(auto_trace.ImageProcessorDialog):
    def __init__(self, boundary, layers, img, command):
        super().__init__(boundary, layers, img, command)

        self.rect = [2130, 953, 2184, 1234]
        self.dragged = True
        self.points = []

        self.canvas.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.canvas.bind("<Button-1>", self.clicked)
        self.canvas.canvas.bind("<ButtonRelease-1>", self.released)

        self.released(None)
        # self.update_image()

    def clicked(self, event):
        self.dragged = False

    def add_pin(self, pins, components):
        print(pins)
        print(components)

    def released(self, event):
        print(self.rect)
        if not self.dragged:
            ex, ey = self.canvas.canvas_to_image(event.x_root - self.canvas.winfo_rootx(),
                                                 event.y_root - self.canvas.winfo_rooty())
            if ex >= self.disp_img.shape[1] or ey >= self.disp_img.shape[0]: return

            view_comp.PinCreateDialog([], "", int(ex), int(ey),
                                      event.state & 1, self.add_pin)

            return
        self.dragged = True

        x0, y0, x1, y1 = self.rect
        ker = self.img[y0:y1, x0:x1]
        cv2.imwrite("/tmp/kernel.png", ker)
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        gk = gray[y0:y1, x0:x1]
        print("Calculating brightness 1")
        br = cv2.filter2D(gray, cv2.CV_64F, numpy.ones(ker.shape[:2]))/numpy.sum(gk)
        print("Normalizing 1")
        norm = numpy.uint16(self.img)/br[:, :, None]
        print("SSECONV 1")
        conv = sseconv(norm, ker)
        minconv = cv2.erode(conv, numpy.ones((30, 30)))
        maxconv = cv2.dilate(conv, numpy.ones((30, 30)))
        print(conv.dtype)
        with open("/tmp/conv_raw", "wb") as f: f.write(conv.tobytes())
        cv2.imwrite("/tmp/diff.png", numpy.uint8(255 * (conv - minconv)/(maxconv - minconv)))
        relmin = numpy.nonzero(minconv == conv)
        s = numpy.argsort(conv[relmin[0], relmin[1]])
        self.points = []
        for a, b in zip(relmin[0][s[:100]], relmin[1][s[:100]]):
            # print("{:4} {:4} {:12.0f}".format(a, b, conv[a, b]))
            self.points.append((b, a, conv[a, b]))
        print("Convolution done")
        cv2.imwrite("/tmp/norm.png", numpy.uint8(255*norm/numpy.max(norm)))
        conv = numpy.log10(minconv)
        convimg = numpy.uint8(numpy.clip(255*(conv-numpy.min(conv))/(numpy.max(conv)-numpy.min(conv)), 0, 255))
        convimg = cv2.cvtColor(convimg, cv2.COLOR_GRAY2BGR)
        convimg[relmin[0][s[:100]], relmin[1][s[:100]], 1] = 255
        cv2.imwrite("/tmp/conv.png", convimg)
        plt.yscale("log")
        plt.plot([x[2] for x in self.points])
        plt.show()
        self.update_image()

    def drag(self, event):
        ex, ey = self.canvas.canvas_to_image(event.x_root - self.canvas.winfo_rootx(),
                                             event.y_root - self.canvas.winfo_rooty())
        if ex >= self.disp_img.shape[1] or ey >= self.disp_img.shape[0]: return
        if not self.dragged:
            self.rect = [int(ex), int(ey), 0, 0]
            self.dragged = True
        self.rect[2] = int(ex)
        self.rect[3] = int(ey)
        self.update_image()

    def update_image(self):
        self.disp_img = self.img.copy()
        if self.rect is not None:
            x0, y0, x1, y1 = self.rect
            self.rect = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
            x0, y0, x1, y1 = self.rect
            cv2.rectangle(self.disp_img, self.rect[:2], self.rect[2:], (0, 0, 255), 10)
            n = 255
            for px, py, v in self.points:
                cv2.rectangle(self.disp_img, (px - (x1-x0)//2, py - (y1-y0)//2),
                              (px - (x1-x0)//2 + (x1-x0), py - (y1-y0)//2 + (y1-y0)), (0, n, 255-n), 8)
                n = max(0, n-10)
        self.canvas.update_image()

    def finished(self):
        self.command(self.points)


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
    with zipfile.ZipFile("/Users/matthias/Documents/reverse_engineering/dsi.zip") as z:
        solder = cv2.imdecode(numpy.frombuffer(z.read("solder.png"), numpy.uint8), cv2.IMREAD_COLOR)
        comp = cv2.imdecode(numpy.frombuffer(z.read("comp.png"), numpy.uint8), cv2.IMREAD_COLOR)
        bottom = cv2.imdecode(numpy.frombuffer(z.read("LAYERBottom.png"), numpy.uint8), cv2.IMREAD_COLOR)
        bound = layer.Layer("__boundary__", z.read("boundary_layer.png"))
    print(solder.shape)
    out = layer.Layer("test", shape=solder.shape)
    FindComponentsDialog(bound, [out], [comp, solder], print)
    tk.mainloop()
    exit()

if __name__ == '__main__':
    img = cv2.imread("/tmp/cnc/comp.png")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x0, y0, x1, y1 = 2245, 966, 2245+45, 966+32
    # ker = img[1100:1354, 798:798+50]
    # gk = gray[1100:1354, 798:798+50]
    # ker = img[1048:1048+263, 2366:2366+68]
    # gk = gray[1048:1048+263, 2366:2366+68]
    ker = img[y0:y1, x0:x1]
    gk = gray[y0:y1, x0:x1]
    print("Calculating brightness 1")
    br = cv2.filter2D(gray, cv2.CV_32F, numpy.ones(ker.shape[:2]))/numpy.sum(gk)
    print("Normalizing 1")
    norm = img/br[:, :, None]
    print("SSECONV 1")
    conv = sseconv(norm, ker)

    """print("Calculating brightness 2")
    ker2 = ker.transpose((1, 0, 2))[:, ::-1]
    gk2 = gk.transpose()[:, ::-1]
    br2 = cv2.filter2D(gray, cv2.CV_32F, numpy.ones(ker2.shape[:2])) / numpy.sum(gk2)
    print("Normalizing 2")
    norm2 = img / br2[:, :, None]
    print("SSECONV 2")
    conv2 = sseconv(norm2, ker2)"""
    print("done")


    img2 = img.copy()
    flat = conv.flatten()
    m = numpy.argsort(flat)[:500]
    points = []
    for i in m:
        y, x = divmod(i, img.shape[1])
        for p in points:
            if p[0]-10 <= x < p[0]+10 and p[1]-10 <= y <= p[1]+10:
                p[2] = min(p[2], flat[i])
                break
        else:
            points.append([x, y, flat[i]])
            print((x, y), flat[i])
            cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
    print(points)

    """flat2 = conv2.flatten()
    m = numpy.argsort(flat2)[:2000]
    points = []
    for i in m:
        y, x = divmod(i, img.shape[1])
        for p in points:
            if p[0] - 10 <= x < p[0] + 10 and p[1] - 10 <= y <= p[1] + 10:
                p[2] = min(p[2], flat2[i])
                break
        else:
            points.append([x, y, flat2[i]])
            print((x, y), flat2[i])
            cv2.circle(img2, (x, y), 10, (0, 0, 255), -1)
    print(points)"""

    cv2.imshow("img", img)
    # cv2.imshow("conv", numpy.uint8(80*numpy.log10(conv)))
    cv2.imshow("conv", numpy.uint8(numpy.clip(255*conv/4e6, 0, 255)))
    cv2.imshow("br", numpy.uint8(255*br/numpy.max(br)))
    cv2.imshow("normalized", numpy.uint8(norm))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """cv2.imshow("img2", img2)
    cv2.imshow("conv2", numpy.uint8(numpy.clip(255*conv2/1e8, 0, 255)))
    cv2.imshow("br2", numpy.uint8(255*br2/numpy.max(br2)))
    cv2.imshow("normalized2", numpy.uint8(norm2))
    cv2.waitKey(0)"""

    X, Y = numpy.meshgrid(numpy.arange(250), numpy.arange(254))
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, conv[1100:1354, 798-200:798+50])
    ax.set_zscale("log")
    plt.show()
