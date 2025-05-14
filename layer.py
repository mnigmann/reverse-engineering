import time

try:
    from cv2 import cv2
except ImportError:
    import cv2
import numpy


class Layer:
    NETLIST = 0b1000
    VISIBLE = 0b0100
    MODE_SELECT = 0b0000
    MODE_DRAW = 0b0001
    MODE_ERASE = 0b0010
    MODE_PIN = 0b0011

    def __init__(self, name, image=None, shape=None):
        self.color = [255, 0, 0]
        self.thickness = 1
        self.mode = 0
        self.nets_changed = True
        if image is not None:
            self.layer = self._bytes2img(image)
            if self.layer.shape[2] == 3:
                # First pixel  [R,  layer0,   layer1]
                # Second pixel [G,  layer0,   layer1]
                # Third pixel  [B,  layer0,   layer1]
                print("Control pixels for {}: ".format(name), self.layer[0:2, 0:6, 0])
                self.nets = numpy.array(self.layer, dtype=numpy.uint16)
                self.nets = self.nets[:, :, 1] | (self.nets[:, :, 2] << 8)
                self.thickness = (self.layer[1, 0, 0] >> 2)
                self.mode = (self.layer[1, 1, 0] >> 2) & 0b00001100
                self.color = list(self.layer[0, 0:3, 0])
                self.layer[0, 0:3, 0] = self.layer[0, 3:6, 0] >> 2
                print("loading layer", self.thickness, self.mode, self.color)
                self.layer[:, :, 1:2] = 0
                self.layer[self.layer[:, :, 0] == 1] = [1, 1, 1]
            elif self.layer.shape[2] == 4:
                # First row, alpha channel:
                #   R   G   B   thick   mode
                self.color = list(self.layer[0, 0:3, 3])
                self.thickness = self.layer[0, 3, 3]
                self.mode = self.layer[0, 4, 3] & 0b00001100
                self.layer[0, 0:5, 3] = 0
                self.nets = numpy.uint32(self.layer)
                self.nets = self.nets[:, :, 0] | (self.nets[:, :, 1] << 8) | (self.nets[:, :, 2] << 16) | (self.nets[:, :, 3] << 24)
                #self.layer = numpy.tensordot(numpy.where(self.nets, numpy.uint8(1), numpy.uint8(0)), numpy.uint8([1, 1, 1]), 0)
                self.layer = numpy.where(self.nets, numpy.uint8(1), numpy.uint8(0))
                self.layer = numpy.ascontiguousarray(numpy.transpose([self.layer, self.layer, self.layer], (1, 2, 0)))
        elif shape:
            self.layer = numpy.zeros(shape, dtype=numpy.uint8)
            self.nets = numpy.zeros(shape[:2], dtype=numpy.uint16)
        self.mapped = numpy.zeros(65536, dtype=numpy.uint16)        # mapping[netnum] = globally unique
        self.rmap = numpy.zeros(65536, dtype=numpy.uint16)          # rmap[globally unique] = netnum
        # self.layer[:200, :200, :] = 1
        self.fit = self.layer
        self.name = name
        self.last_pos = (0, 0)

        self.dim_mask = None
        self.fit_dim = None

    def _bytes2img(self, b):
        return cv2.imdecode(numpy.frombuffer(b, numpy.uint8), cv2.IMREAD_UNCHANGED)

    def apply(self, img):
        if self.mode & 0b00000100:
            if self.fit_dim is not None:
                return numpy.where(self.fit, numpy.full(img.shape, (self.fit-0.5*self.fit_dim)*numpy.array(self.color), dtype=numpy.uint8), img)
            else:
                return numpy.where(self.fit, numpy.full(img.shape, self.fit * numpy.array(self.color), dtype=numpy.uint8), img)
        else: return img

    def motion(self, x, y):
        # if drawing...
        if self.mode & 0b11 == 0b01:
            print(self.layer.shape, self.layer.dtype)
            self.layer = cv2.line(self.layer, self.last_pos, (x, y), [1, 1, 1], self.thickness)
            self.last_pos = (x, y)
            return True
        # if erasing...
        if self.mode & 0b11 == 0b10:
            self.layer = cv2.line(self.layer, self.last_pos, (x, y), [0, 0, 0], self.thickness)
            self.last_pos = (x, y)
            return True
        return False

    def click(self, x, y, mode):
        if self.name == "__component__": print("clicked at {}, {} and color is {}".format(x, y, self.color))
        self.mode = (self.mode & 0b11111100) | (mode & 0b00000011)
        self.last_pos = (x, y)
        self.motion(x, y)
        self.nets_changed = True

    def unclick(self):
        self.mode &= 0b11111100

    def dumps(self):
        nets = numpy.where(self.nets, self.nets, self.layer[:, :, 0])*self.layer[:, :, 0]
        img = numpy.zeros(nets.shape+(4,))
        img[:, :, 0] = nets & 0xff; nets = nets >> 8
        img[:, :, 1] = nets & 0xff; nets = nets >> 8
        img[:, :, 2] = nets & 0xff; nets = nets >> 8
        img[:, :, 3] = nets & 0xff
        img[0, 0:3, 3] = self.color
        img[0, 3, 3] = self.thickness
        img[0, 4, 3] = self.mode
        """
        img = self.layer.copy()
        # Row 0, red channel: R,       G,       B, 3/0, 4/1, 5/2
        # Row 1, red channel: 0/thick, 1/mode,  2, 3,   4,   5
        img[:, :, 1] = (self.nets & 255)
        img[:, :, 2] = (self.nets >> 8)
        img[0, 3, 0] |= (img[0, 0, 0] << 2)
        img[0, 4, 0] |= (img[0, 1, 0] << 2)
        img[0, 5, 0] |= (img[0, 2, 0] << 2)
        img[0, 0:3, 0] = self.color
        img[1, 0, 0] = (img[1, 0, 0] & 1) | (self.thickness << 2)
        img[1, 1, 0] = (img[1, 1, 0] & 1) | (self.mode << 2)"""
        print("Dumping {} with thickness {}, state {}".format(self.name, self.thickness, self.mode))
        return cv2.imencode(".png", img)[1].tostring()
