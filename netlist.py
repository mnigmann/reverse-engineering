import math
import os
import re
import tkinter
import networkx
import itertools
import numpy


class NetlistExporter:
    def __init__(self, pins, layers, netnames, components):
        self.connected = {}
        self.connected_full = {}
        self.coords = {}
        self.components = components
        self.pins = pins
        self.layers = {l.name: l for l in layers}
        self.netnames = netnames
        self.nets = {}

        for x in self.pins:
            if not x[2]: continue
            l = self.layers[x[4]]
            if x[2] not in self.connected: self.connected[x[2]] = {}
            if x[2] not in self.connected_full: self.connected_full[x[2]] = []
            nn = l.nets[x[1], x[0]]
            self.connected[x[2]][x[3]] = self.netnames.get(nn, str(nn))
            self.connected_full[x[2]].append([x[0], x[1], x[3], x[3], self.netnames.get(nn, str(nn))])

        print("connected", self.connected)

    def convert_components(self, rules):
        self.raw_connected = {}

        for x, pins in self.connected.items():
            self.raw_connected[x] = pins.copy()
            c = self.connected_full[x]
            pn = [y.lower() for y in pins]
            for r in rules:
                if len(r) == 2: r += (" "*len(r[0]),)
                use = True
                for i, o, d in zip(*r):
                    if i not in pn and d == " ": use = False
                if use:
                    for i, o, d in zip(*r):
                        if d == " ": pins[o] = pins.pop(i.lower(), pins.pop(i.upper()))
                        else: pins[o] = pins.pop(i.lower(), pins.pop(i.upper, pins[d]))
                    m = dict(zip(*r[:2]))
                    for y in c: y[3] = m[y[2].lower()]
            """if "b" in pn and "c" in pn and "e" in pn:
                # BJT found
                pins["1"] = pins.pop("c", pins.pop("C"))
                pins["2"] = pins.pop("b", pins.pop("B"))
                pins["3"] = pins.pop("e", pins.pop("E"))
                for y in c: y[3] = {"c": "1", "b": "2", "e": "3"}[y[2].lower()]
            elif "d" in pn and "g" in pn and "s" in pn:
                # MOSFET found
                pins["1"] = pins.pop("d", pins.pop("D"))
                pins["2"] = pins.pop("g", pins.pop("G"))
                pins["3"] = pins.pop("s", pins.pop("S"))
                pins["4"] = pins.pop("b", pins.pop("B", pins["3"]))  # If no body pin, use the source pin
                for y in c: y[3] = {"d": "1", "g": "2", "s": "3"}[y[2].lower()]
                # c["1"] = c.pop("d", c.pop("D"))
                # c["2"] = c.pop("g", c.pop("G"))
                # c["3"] = c.pop("s", c.pop("S"))
                # c["4"] = c.pop("b", c.pop("B", c["3"]))
            elif "a" in pn and "g" in pn and "k" in pn:
                pins["1"] = pins.pop("k", pins.pop("K"))
                pins["2"] = pins.pop("g", pins.pop("G"))
                pins["3"] = pins.pop("a", pins.pop("A"))
                for y in c: y[3] = {"k": "1", "g": "2", "a": "3"}[y[2].lower()]
                # c["1"] = c.pop("k", c.pop("K"))
                # c["2"] = c.pop("g", c.pop("G"))
                # c["3"] = c.pop("a", c.pop("A"))
            elif "a" in pn and "k" in pn:
                # Diode found
                pins["1"] = pins.pop("a", pins.pop("A"))
                pins["2"] = pins.pop("k", pins.pop("K"))
                for y in c: y[3] = {"a": "1", "k": "2"}[y[2].lower()]
                # c["1"] = c.pop("a", c.pop("A"))
                # c["2"] = c.pop("k", c.pop("K"))"""

        for comp, pins in self.connected.items():
            for p, n in pins.items():
                if n not in self.nets: self.nets[n] = []
                self.nets[n].append((comp, p))
        print("self.nets", self.nets)

    def save_to_file(self, fname):
        pass


class SpiceNetlistExporter(NetlistExporter):
    def __init__(self, pins, layers, netnames, components):
        super().__init__(pins, layers, netnames, components)

    def save_to_file(self, fname, name="unnamed circuit"):
        self.convert_components([
            ("cbe", "123"),
            ("dgsb", "1234", "   3"),
            ("kga", "123"),
            ("ak", "12")
        ])
        with open(fname, "w") as circ:
            circ.write(name+"\n")
            for x, pins in self.connected.items():
                circ.write(x + " " + " ".join(y[1] for y in sorted(pins.items(), key=lambda x: int(x[0]))))
                if x in self.components:
                    comp = self.components[x]
                    val = [y[1] for y in comp if y[0] == "value"]
                    if val: circ.write(" " + str(val[0]))
                    else: circ.write(" <no value>")
                circ.write("\n")
            circ.write(".END\n")


class PCBNetlistExporter(NetlistExporter):
    def __init__(self, pins, layers, netnames, components):
        super().__init__(pins, layers, netnames, components)

    def save_to_file(self, fname, name="unnamed circuit"):
        self.convert_components([("ak", "12")])
        nets = {}
        for refdes, pins in self.connected.items():
            for p, n in pins.items():
                if n not in nets: nets[n] = ""
                nets[n] += refdes+'-'+p+' '
        with open(fname, "w") as f:
            for n, v in nets.items():
                f.write(n+"\t"+v[:-1]+"\n")


class GschemExporter(NetlistExporter):
    def __init__(self, pins, layers, netnames, components):
        super().__init__(pins, layers, netnames, components)
        self.spread = 8
        self.symbols = [
            "/Users/matthias/Downloads/geda-gaf-1.10.2/symbols/",
            "/Users/matthias/Documents/symbols"
        ]
        self.convert_components([
            ("cbe", "123"),
            ("dgsb", "1234", "   3"),
            ("kga", "123"),
            ("ak", "12")
        ])
        self.net_points = {x: [] for x in self.nets}
        self.symbol_pins = {}

    def gen_component_text(self, x, y, angle, mirror, file, attr, setup):
        res = "C {} {} 1 {} {} {}\n{{\n".format(int(x), int(y), int(angle), mirror, file)
        rmat = ({
            0:   [1, 0, 0, 1],
            90:  [0, -1, 1, 0],
            180: [-1, 0, 0, -1],
            270: [0, 1, -1, 0]
        })[angle]
        for k, v in attr:
            if k not in setup: continue
            s = setup[k]
            res += "T {} {} 5 10 1 1 {} 0 1\n{}={}\n".format(x+rmat[0]*s[0]+rmat[1]*s[1], y+rmat[2]*s[0]+rmat[3]*s[1], angle, k, v)
        res += "}\n"
        return res

    def open_symbol(self, symname, *args, **kwargs):
        for start in self.symbols:
            fn = self.locate_file(start, symname)
            if fn is None: continue
            return open(fn, *args, **kwargs)
        print("Failed to find", symname)

    def locate_file(self, start, symname):
        ps = os.path.join(start, symname)
        if os.path.isfile(ps) and "gnetman" not in ps:
            return ps
        for x in os.listdir(start):
            px = os.path.join(start, x)
            if os.path.isdir(px):
                rec = self.locate_file(px, symname)
                if rec is not None: return rec

    def get_pin(self, pins, pinlabel=None, pinnumber=None):
        for p in pins:
            if p[2] == pinlabel: return p
        for p in pins:
            if p[3] == pinnumber: return p
        print("Failed to find pinlabel={}, pinnumber={} in {}".format(pinlabel, pinnumber, pins))

    def parse_component(self, file):
        if file not in self.symbol_pins:
            sym_pins = []
            with self.open_symbol(file) as f:
                in_attr = False
                head = ""
                for l in f:
                    if l[0] == "{":
                        in_attr = True
                        attr = ""
                    elif l[0] == "}":
                        if head[0] == "P":
                            l = head.split(" ")
                            label = re.search("^pinlabel=(.+)", attr, re.MULTILINE)
                            number = re.search("^pinnumber=(.+)", attr, re.MULTILINE)
                            sym_pins.append([int(l[1+2*int(l[7])]), int(l[2+2*int(l[7])]),
                                             (label.group(1) if label else number.group(1)),
                                             (number.group(1) if number else label.group(1))])
                        in_attr = False
                    elif in_attr: attr += l
                    else: head = l
            print(file, "has pins", sym_pins)
            self.symbol_pins[file] = sym_pins
        else: sym_pins = self.symbol_pins[file]
        return sym_pins

    def add_general_component(self, file, pins, attr, setup, angle=0, flip=0, offset=(0,0)):
        flip = -1 if flip else 1
        sym_pins = self.parse_component(file)

        rmat = ({
            0: [1, 0, 0, 1],
            90: [0, -1, 1, 0],
            180: [-1, 0, 0, -1],
            270: [0, 1, -1, 0]
        })[angle]
        # The user-inserted pins will almost always be a subset of the symbol's pins
        x, y = self.get_pin(pins, pinnumber="1")[:2]
        for p in pins:
            sp = self.get_pin(sym_pins, p[2], p[3])
            if sp is None: continue
            coords = [round(self.spread*x+offset[0], -2)+rmat[0]*sp[0]*flip+rmat[1]*sp[1], round(self.spread*y+offset[1], -2)+rmat[2]*sp[0]*flip+rmat[3]*sp[1]]
            #print(sym_pins, p, coords)
            self.net_points[p[4]].append(coords)

    def gen_net_points(self):
        res = ""
        for n, x in self.net_points.items():
            if len(x) <= 1: continue
            print("net {} with points {}".format(n, x), end="")
            g = networkx.Graph()
            g.add_weighted_edges_from([(a, b, math.sqrt((x[a][0]-x[b][0])**2+(x[a][1]-x[b][1])**2)) for a, b in itertools.combinations(range(len(x)), 2)])
            ne = networkx.approximation.traveling_salesman_problem(g, cycle=False)
            print("goes in order {}".format(ne))
            for i in range(1, len(ne)):
                res += "N {} {} {} {} 4\n".format(*x[ne[i-1]][:2], *x[ne[i]][:2])
        return res

    def save_to_file(self, fname, name="unnamed circuit"):
        cont = "v 20201216 2\n"
        for c in self.components:
            if c not in self.connected_full: continue
            p = self.connected_full[c]
            f = [x[1] for x in self.components[c] if x[0] == "symbol"]
            base = c.strip("0123456789")
            print("inserting component", c, base, self.components[c])
            if base == "R" and not f:
                x, y = self.get_pin(p, pinnumber="1")[:2]
                x2, y2 = self.get_pin(p, pinnumber="2")[:2]
                angle = round((2*math.atan2(y2-y, x2-x)/math.pi))*90 % 360
                self.add_general_component("resistor-1.sym", self.connected_full[c], self.components[c], {}, angle)
                cont += self.gen_component_text(round(x*self.spread, -2), round(y*self.spread, -2), angle, 0, "resistor-1.sym", self.components[c], {
                    "refdes": [0, 200, 0, 1],
                    "value": [500, 200, 0, 1]
                })
            elif base == "C" and not f:
                x, y = self.get_pin(p, pinnumber="1")[:2]
                x2, y2 = self.get_pin(p, pinnumber="2")[:2]
                angle = round((2*math.atan2(y2-y, x2-x)/math.pi))*90 % 360
                self.add_general_component("capacitor-1.sym", self.connected_full[c], self.components[c], {}, angle)
                cont += self.gen_component_text(round(x*self.spread, -2), round(y*self.spread, -2), angle, 0, "capacitor-1.sym", self.components[c], {
                    "refdes": [0, 300, 0, 1],
                    "value": [500, 300, 0, 1]
                })
            elif base == "L" and not f:
                x, y = self.get_pin(p, pinnumber="1")[:2]
                x2, y2 = self.get_pin(p, pinnumber="2")[:2]
                angle = round((2 * math.atan2(y2 - y, x2 - x) / math.pi)) * 90 % 360
                self.add_general_component("inductor-1.sym", self.connected_full[c], self.components[c], {}, angle)
                cont += self.gen_component_text(round(x * self.spread, -2), round(y * self.spread, -2), angle, 0,
                                                "inductor-1.sym", self.components[c], {
                                                    "refdes": [0, 200, 0, 1],
                                                    "value": [500, 200, 0, 1]
                                                })
            elif base == "D" and not f:
                x, y = self.get_pin(p, pinnumber="1")[:2]
                x2, y2 = self.get_pin(p, pinnumber="2")[:2]
                angle = round((2*math.atan2(y2-y, x2-x)/math.pi))*90 % 360
                #print(c, "is at", x, y, x2, y2, angle)
                self.add_general_component("diode-1.sym", self.connected_full[c], self.components[c], {}, angle)
                cont += self.gen_component_text(round(x*self.spread, -2), round(y*self.spread, -2), angle, 0, "diode-1.sym", self.components[c], {
                    "refdes": [0, 300, 0, 1],
                })
            elif base == "Q" and f:
                x, y = self.get_pin(p, pinnumber="1")[:2]
                self.add_general_component(f[0], self.connected_full[c], self.components[c], {})
                cont += self.gen_component_text(round(x*self.spread, -2), round(y*self.spread, -2), 0, 0, f[0], self.components[c], {
                    "refdes": [300, 500, 0, 1],
                })
            elif base == "U" and f:
                x, y = self.get_pin(p, pinnumber="1")[:2]
                pins = self.connected_full[c]
                self.parse_component(f[0])
                v_pcb = numpy.zeros((2, 2))
                v_sym = numpy.zeros((2, 2))
                m = pins[0]
                orig = [0, 0]
                for p in pins:
                    if int(p[3]) > int(m[3]): m = p
                    if p[3] == "1":
                        v_pcb[0, 0] -= p[0]; v_pcb[0, 1] -= p[0]; v_pcb[1, 0] -= p[1]; v_pcb[1, 1] -= p[1]
                    if p[3] == "2":
                        v_pcb[0, 1] += p[0]; v_pcb[1, 1] += p[1]
                v_pcb[0, 0] += m[0]; v_pcb[1, 0] += m[1]
                m = self.symbol_pins[f[0]][0]
                for p in self.symbol_pins[f[0]]:
                    if int(p[3]) > int(m[3]): m = p
                    if p[3] == "1":
                        orig = p[:2]
                        v_sym[0, 0] -= p[0]; v_sym[0, 1] -= p[0]; v_sym[1, 0] -= p[1]; v_sym[1, 1] -= p[1]
                    if p[3] == "2":
                        v_sym[0, 1] += p[0]; v_sym[1, 1] += p[1]
                v_sym[0, 0] += m[0]; v_sym[1, 0] += m[1]
                # m_rot @ m_ref @ v_sym = v_pcb
                # m_rot @ m_ref = v_pcb @ v_sym^-1
                m_inv = v_pcb @ numpy.linalg.inv(v_sym)
                flip = numpy.linalg.det(m_inv) < 0
                if flip:
                    m_inv[:, 0] *= -1
                    print("flip, ", end="")
                if numpy.linalg.det(abs(m_inv)) < 0:
                    if m_inv[1, 0] < 0: angle = 270
                    else: angle = 90
                else:
                    if m_inv[0, 0] < 0: angle = 180
                    else: angle = 0
                print("rotate", angle)
                rmat = ({
                    0: [1, 0, 0, 1],
                    90: [0, -1, 1, 0],
                    180: [-1, 0, 0, -1],
                    270: [0, 1, -1, 0]
                })[angle]
                if flip: orig[0] = -orig[0]
                orig = [-rmat[0]*orig[0]-rmat[1]*orig[1], -rmat[2]*orig[0]-rmat[3]*orig[1]]
                self.add_general_component(f[0], pins, self.components[c], {}, angle, flip, offset=orig)
                cont += self.gen_component_text(round(x*self.spread+orig[0], -2), round(y*self.spread+orig[1], -2), angle, int(flip), f[0], self.components[c], {
                    "refdes": [0, 0, 0, 1]
                })
            elif f:
                x, y = self.get_pin(p, pinnumber="1")[:2]
                self.add_general_component(f[0], self.connected_full[c], self.components[c], {})
                cont += self.gen_component_text(round(x * self.spread, -2), round(y * self.spread, -2), 0, 0, f[0],
                                                self.components[c], {
                                                    "refdes": [300, 500, 0, 1],
                                                })
            else:
                print("SKIPPING COMPONENT", c)


        cont += self.gen_net_points()
        with open(fname, "w") as f:
            f.write(cont)


