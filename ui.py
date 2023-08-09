import json
import tkinter
import zipfile
from tkinter import filedialog
from tkinter import colorchooser
from tkinter import simpledialog
from cv2 import cv2
import numpy
import os.path
import time

import fill_holes
import find_components
import list_components
import netlist
import view_comp
import scrollable_image
import auto_trace
import find_loops
import layer
import newproject
import projectsetup

import matplotlib.pyplot as plt
import networkx


class MainWindow(tkinter.Tk):
    def __init__(self):
        super().__init__()
        self.photo = []
        self.view_box = []
        self.solder_arr = None
        self.comp_arr = None
        self.combined_arr = None
        self.solder_fit = None
        self.comp_fit = None
        self.solder_id = None
        self.comp_id = None
        self.canvas_lastsize = []
        self.hflip = False

        self.project = {}
        self.layers = []
        self.component_layer = None
        self.boundary_layer = None
        self.pname = ""

        self.opacity = 0

        self.selected_net = None

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.menubar = tkinter.Menu(self)
        self.filemenu = tkinter.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="New", command=self.new_project)
        self.filemenu.add_command(label="Open", command=self.open_project_dialog)
        self.filemenu.add_command(label="Save", command=self.save_project)
        self.menubar.add_cascade(label="File", menu=self.filemenu)

        self.editmenu = tkinter.Menu(self.menubar, tearoff=0)
        self.editmenu.add_command(label="Fill boundary", command=self.fill_boundary)
        self.editmenu.add_separator()
        self.editmenu.add_command(label="New layer", command=self.new_layer)
        self.editmenu.add_command(label="Export layer", command=self.export_layer)
        self.editmenu.add_separator()
        self.editmenu.add_command(label="Dilate", command=self.transform_dilate)
        self.editmenu.add_command(label="Erode", command=self.transform_erode)
        self.editmenu.add_command(label="Sobel", command=self.transform_sobel)
        self.editmenu.add_command(label="Fill holes", command=self.transform_fill_holes)
        self.editmenu.add_command(label="block avg", command=self.transform_block_average)
        self.editmenu.add_separator()
        self.editmenu.add_command(label="List components", command=lambda: list_components.ComponentListDialog(self.project["components"], self.project["pins"], self.scrollto))
        self.editmenu.add_command(label="Find net", command=self.find_net)
        self.menubar.add_cascade(label="Edit", menu=self.editmenu)

        self.tracemenu = tkinter.Menu(self.menubar, tearoff=0)
        self.tracemenu.add_command(label="Threshold",
                                   command=lambda: auto_trace.ThresholdDialog(self.boundary_layer, self.layers,
                                                                                    [self.solder_arr, self.comp_arr],
                                                                                    self.canvas.update_image))

        self.tracemenu.add_command(label="Color distance threshold",
                                   command=lambda: auto_trace.ColorDistanceThresholdDialog(self.boundary_layer, self.layers,
                                                                                    [self.solder_arr, self.comp_arr],
                                                                                    self.canvas.update_image))
        self.tracemenu.add_command(label="Color gradient distance threshold",
                                   command=lambda: auto_trace.GradientThresholdDialog(self.boundary_layer, self.layers,
                                                                                    [self.solder_arr, self.comp_arr],
                                                                                    self.canvas.update_image))
        self.tracemenu.add_command(label="Triangulation interpolation distance threshold",
                                   command=lambda: auto_trace.DelaunayThresholdDialog(self.boundary_layer, self.layers,
                                                                                      [self.solder_arr, self.comp_arr],
                                                                                      self.canvas.update_image))
        self.tracemenu.add_separator()
        self.tracemenu.add_command(label="Identify components",
                                   command=lambda: find_components.FindComponentsDialog(self.boundary_layer, self.layers,
                                                                                        [self.solder_arr, self.comp_arr],
                                                                                        self.on_components_identified))
        self.tracemenu.add_separator()
        self.tracemenu.add_command(label="Check circuit", command=self.check_circuit)
        self.tracemenu.add_command(label="Generate netlist", command=self.generate_netlist)
        self.tracemenu.add_command(label="Export SPICE netlist", command=lambda: self.export_netlist("spice"))
        self.tracemenu.add_command(label="Export PCB netlist", command=lambda: self.export_netlist("pcb"))
        self.tracemenu.add_command(label="Export gEDA schematic", command=lambda: self.export_netlist("gschem"))
        self.menubar.add_cascade(label="Auto trace", menu=self.tracemenu)

        self.contextmenu = tkinter.Menu(self, tearoff=0)


        # Configure the drawing area
        self.view_frame = tkinter.Frame(self)
        self.view_frame.columnconfigure(0, weight=1)
        self.view_frame.rowconfigure(0, weight=1)

        self.canvas = scrollable_image.ScrollableImage(self, width=400, height=400, command=self.on_scroll)
        self.canvas.grid(row=0, column=1, sticky="news")

        self.sidebar = tkinter.Frame(self)
        self.sidebar.grid(row=0, column=0, sticky=tkinter.N)


        # Configure the options for selecting the opacity of the images
        self.opacity_frame = tkinter.LabelFrame(self.sidebar, text="Image selection")
        self.opacity_frame.grid(row=0, column=0, sticky="ew")
        self.opacity_frame.columnconfigure(1, weight=1)

        tkinter.Label(self.opacity_frame, text="Opacity").grid(row=0, column=0)
        self.opacity_entry = tkinter.Entry(self.opacity_frame, width=3)
        self.opacity_entry.grid(row=0, column=1, sticky=tkinter.W)
        self.opacity_entry.insert(0, "0")
        self.opacity_entry.bind("<KeyRelease>", self.update_opacity)

        self.opacity_scale = tkinter.Scale(self.opacity_frame, length=100, orient=tkinter.HORIZONTAL, from_=0, to=256,
                                           showvalue=0, command=self.update_scale)
        self.opacity_scale.grid(row=1, column=0, columnspan=2, sticky=tkinter.N)

        tkinter.Label(self.opacity_frame, text="Preset 1").grid(row=2, column=0)
        tkinter.Label(self.opacity_frame, text="Preset 2").grid(row=3, column=0)
        self.opacity_p1 = tkinter.Entry(self.opacity_frame, width=3)
        self.opacity_p1.grid(row=2, column=1, sticky=tkinter.W)
        self.opacity_p1.insert(0, "0")
        self.opacity_p2 = tkinter.Entry(self.opacity_frame, width=3)
        self.opacity_p2.grid(row=3, column=1, sticky=tkinter.W)
        self.opacity_p2.insert(0, "256")


        # Configure the drawing layers
        self.layers_frame = tkinter.LabelFrame(self.sidebar, text="Drawing layers")
        self.layers_frame.grid(row=1, column=0, sticky="ew")

        self.layers_list = tkinter.Listbox(self.layers_frame, width=15, selectmode=tkinter.SINGLE, height=5, exportselection=False)
        self.layers_list.grid(row=0, column=0, sticky="ew")


        # Configure the drawing options
        self.draw_frame = tkinter.LabelFrame(self.sidebar, text="Drawing options")
        self.draw_frame.grid(row=2, column=0, sticky="ew")
        self.draw_frame.columnconfigure(1, weight=1)
        tkinter.Label(self.draw_frame, text="Color").grid(row=0, column=0)
        tkinter.Label(self.draw_frame, text="Thickness").grid(row=1, column=0)
        self.layers_color = tkinter.Frame(self.draw_frame, bg="black")
        self.layers_color.grid(row=0, column=1, sticky="news")
        self.layers_color.bind("<ButtonRelease>", lambda x: self.set_color())
        self.layers_thick = tkinter.Entry(self.draw_frame, width=3, validate="key", validatecommand=(self.register(lambda x: x.isdecimal() or not x), "%P"))
        self.layers_thick.grid(row=1, column=1, sticky="w")
        self.layers_thick.insert(0, "1")
        self.layers_thick.bind("<KeyRelease>", self.apply_thickness)
        self.layers_thick.bind("<d>", lambda x: self.draw_mode.set(1))
        self.layers_thick.bind("<e>", lambda x: self.draw_mode.set(2))

        self.draw_mode = tkinter.IntVar()
        tkinter.Radiobutton(self.draw_frame, text="Select", variable=self.draw_mode, value=0).grid(row=2, column=0,
                                                                                                   columnspan=2,
                                                                                                   sticky="w")
        tkinter.Radiobutton(self.draw_frame, text="Draw", variable=self.draw_mode, value=1).grid(row=3, column=0,
                                                                                                 columnspan=2,
                                                                                                 sticky="w")
        tkinter.Radiobutton(self.draw_frame, text="Erase", variable=self.draw_mode, value=2).grid(row=4, column=0,
                                                                                                  columnspan=2,
                                                                                                  sticky="w")
        tkinter.Radiobutton(self.draw_frame, text="Add pin", variable=self.draw_mode, value=3).grid(row=5, column=0,
                                                                                                    columnspan=2,
                                                                                                    sticky="w")

        # Configure the label for displaying the net
        self.net_label = tkinter.Label(self.sidebar, text="netname: ")
        self.net_label.grid(row=3, column=0)
        self.pin_label = tkinter.Label(self.sidebar, text="refdes:\npinnumber:")
        self.pin_label.grid(row=4, column=0)

        self.config(menu=self.menubar)
        self.update()
        self.bind("<Motion>", self.propagate_motion)
        self.bind("<Tab>", self.toggle_presets)
        self.canvas.bind("<Button-1>", self.propagate_click)
        self.canvas.bind("<ButtonRelease>", self.propagate_unclick)
        self.canvas.bind("<Double-Button-1>", self.edit_component)
        self.canvas.canvas.bind("<Configure>", lambda x: self.canvas.update_image())
        self.bind_all("<Control-s>", lambda x: self.save_project())
        self.bind_all("<Control-l>", lambda x: self.hide_show_layer())
        self.bind_all("<Control-h>", self.hflip_t)
        self.bind_all("<Control-v>", self.vflip_t)
        self.bind_all("<Escape>", lambda x: self.draw_mode.set(0))
        self.layers_list.bind("<Button-2>", self.show_context_menu)
        self.layers_list.bind("<<ListboxSelect>>", lambda x: self.update_color())

    def _bytes2img(self, b):
        return cv2.imdecode(numpy.frombuffer(b, numpy.uint8), cv2.IMREAD_COLOR)

    def _arr2color(self, a):
        return "#" + "".join(hex(x)[2:].rjust(2, "0") for x in a[::-1])

    def get_active_layer(self) -> layer.Layer:
        selected = self.layers_list.curselection()
        if not selected: return
        selected = self.layers_list.get(selected[0])
        return [l for l in self.layers if l.name == selected][0]

    # Functions for creating new projects
    def new_project(self):
        self._np = newproject.NewProjectDialog(self.on_new_files_chosen)

    def on_new_files_chosen(self, solder, comp, name):
        pname = os.path.splitext(name)[0]+".zip"
        self._ps = projectsetup.ProjectSetupDialog(solder, comp, pname, lambda: self.open_project(pname))

    # Functions for opening a project
    def open_project_dialog(self):
        self.open_project(filedialog.askopenfilename(filetypes=[("Project file", "*.zip")]))

    def open_project(self, pname):
        self.layers_list.delete(0, "end")
        self.layers = []
        self.selected_net = None
        self.pname = pname
        self.boundary_layer = None
        self.component_layer = None
        self.layers_list.insert(0, "__boundary__")
        self.layers_list.insert(0, "__component__")
        n = 2
        self.project = {"pins": [], "components": {}, "nets": {}, "version": "2"}
        with zipfile.ZipFile(pname, "r") as z:
            z.printdir()
            self.solder_arr = self._bytes2img(z.read("solder.png"))
            self.comp_arr = self._bytes2img(z.read("comp.png"))
            for l in z.filelist:
                if l.filename in ["solder.png", "comp.png", "nets.png"]: continue
                if l.filename == "project.json":
                    self.project = json.loads(z.read(l.filename).decode())
                    if "version" not in self.project: self.project["version"] = "1"
                    if self.project.get("version", '1') == '2':
                        self.project["nets"] = {int(n): e for n, e in self.project["nets"].items()}
                if l.filename == "component_layer.png":
                    print("Loading component layer")
                    self.component_layer = layer.Layer("__component__", z.read(l.filename))
                if l.filename == "boundary_layer.png":
                    print("Loading boundary layer")
                    self.boundary_layer = layer.Layer("__boundary__", z.read(l.filename))
                if l.filename.startswith("LAYER"):
                    name = l.filename[5:-4]
                    print("Loading layer", name)
                    cont = z.read(l.filename)
                    print("loaded")
                    self.layers.append(layer.Layer(name, cont))
                    self.layers_list.insert(n, name)
                    n += 1
            if self.boundary_layer is None:
                self.boundary_layer = layer.Layer("__boundary__", shape=self.solder_arr.shape)
            if self.component_layer is None:
                self.component_layer = layer.Layer("__component__", shape=self.solder_arr.shape)
                self.component_layer.mode = layer.Layer.VISIBLE
            if "pins" not in self.project: self.project["pins"] = []
            if "components" not in self.project: self.project["components"] = {}
            print(self.project)
            self.layers.insert(0, self.boundary_layer)
            self.layers.insert(0, self.component_layer)
            self.canvas.set_size(*self.solder_arr.shape[1::-1])
            self.canvas.update_image()
            #self.generate_netlist(False, False)

    def save_project(self):
        print(self.project)
        with open(self.pname, "rb") as f:
            with open(time.strftime("/tmp/backup%H%M%S.zip"), "wb") as o: o.write(f.read())
        with zipfile.ZipFile(self.pname, "w") as z:
            for l in self.layers:
                if l.name == "__boundary__" or l.name == "__component__": continue
                z.writestr("LAYER{}.png".format(l.name), l.dumps())
            z.writestr("component_layer.png", self.component_layer.dumps())
            z.writestr("boundary_layer.png", self.boundary_layer.dumps())
            z.writestr("solder.png", cv2.imencode(".png", self.solder_arr)[1].tobytes())
            z.writestr("comp.png", cv2.imencode(".png", self.comp_arr)[1].tobytes())
            z.writestr("project.json", json.dumps(self.project).encode())

    # Graphics functions
    def on_scroll(self):
        start = time.time()
        self.solder_fit = self.canvas.transform(self.solder_arr)
        self.comp_fit = self.canvas.transform(self.comp_arr)
        merged = cv2.addWeighted(self.solder_fit, 1 - self.opacity, self.comp_fit, self.opacity, 0)
        #print("Image generation took", time.time() - start, self.opacity, merged[20, 20], self.selected_net)
        for l in self.layers[::-1]:
            if l.mode & layer.Layer.VISIBLE:
                l.fit = self.canvas.transform(l.layer)
                if self.selected_net is not None and l.mode & layer.Layer.NETLIST: l.fit_dim = self.canvas.transform(l.dim_mask)
                else: l.fit_dim = None
                merged = l.apply(merged)
        self.canvas.set_image(merged)

    def update_images(self):
        return

    def scrollto(self, point):
        cw, ch = self.canvas.canvas.winfo_width(), self.canvas.canvas.winfo_height()
        print(self.canvas.img_origin, self.canvas.scale, self.canvas.fac, cw, ch)
        scale = self.canvas.scale
        width, height = self.canvas.iw, self.canvas.ih
        self.canvas.img_origin[0] = int(max(0, min(width*(1-1/scale), point[0] - width/scale/2)))
        self.canvas.img_origin[1] = int(max(0, min(height*(1-1/scale), point[1] - height/scale/2)))
        self.canvas.update_image()

    # Function for automatic tracing
    def fill_boundary(self):
        loop_img = self.boundary_layer.layer
        #print(loop_img)
        find_loops.find_loop(loop_img[:, :, 0])
        loop_img[loop_img[:, :, 0] == 1] = [0, 0, 0]
        loop_img[loop_img[:, :, 0] == 2] = [1, 1, 1]
        self.boundary_layer.layer = loop_img
        self.canvas.update_image()

    def generate_netlist(self, recalc=True, genfile=True):
        if "version" in self.project and self.project["version"] == "2":
            self.generate_netlist_v3(recalc, genfile)
        else:
            self.generate_netlist_v1(recalc, genfile)
            # Convert old v1 to new v2/v3
            layers = {l.name: l for l in self.layers}
            for l in self.layers:
                if l.name in ["__component__", "__boundary__"] or not l.mode & layer.Layer.NETLIST: continue
                l.nets = l.mapped[l.nets]

            new_nets = {}
            for net in self.project["nets"]:
                nn, l = net.split(":", 1)
                nn, l = int(nn), layers[l]
                new_nets[int(l.mapped[nn])] = self.project["nets"][net]
                print("{}: {} becomes {}".format(self.project["nets"][net], net, l.mapped[nn]))
            self.project["nets"] = new_nets
            self.project["version"] = "2" #"""

    def generate_netlist_v2(self, recalc=True, genfile=True):
        # Tables needed in new system:
        #   One global table (dict) to map global net numbers to net names (when applicable)
        #   One temporary global table to map the generated numbers to the global net numbers
        #   One temporary global table to map global net numbers to each other
        #   A stack to keep track of global net numbers that are eliminated during netlisting
        # When converting the generated net numbers to global net numbers, they are enumerated
        # to eliminate gaps. The pins are also used to keep track of the numbers so they stay consistent

        layers = {l.name: l for l in self.layers}
        # List of all used global net numbers. First, step through all pins and "lock down" the global
        # net numbers according to the pins.
        used_g = []
        # Next global net number
        next_g = 2
        print("GENERATING NETLIST (v2)", recalc, genfile)
        print(self.project["nets"])
        for l in self.layers:
            if l.name in ["__component__", "__boundary__"] or not l.mode&layer.Layer.NETLIST: continue
            if not recalc: continue
            r = find_loops.find_traces_fast(l.layer[:, :, 0], True, 2)
            unique = numpy.unique(r[0])
            replace = numpy.zeros(len(unique), dtype=numpy.uint32)
            print(l.name, r[1], numpy.max(r[0]), len(numpy.unique(r[0].flatten())))
            new_gdict = {}
            for x in self.project["pins"]:
                if l.name not in x[4:]: continue
                if l.nets[x[1], x[0]] == 0: continue
                prev_g = l.nets[x[1], x[0]]
                print("{}/{} used to be on {}, now on {}".format(x[2], x[3], prev_g, r[0][x[1], x[0]]))
                if prev_g != 1 and prev_g not in used_g:
                    new_gdict[r[0][x[1], x[0]]] = prev_g
                    used_g.append(prev_g)
            used_g.sort()
            print(new_gdict, used_g)
            used_point = 0
            for i, u in enumerate(unique):        # unique is a list of temporary net numbers
                if u == 0: continue
                if u in new_gdict:
                    replace[i] = new_gdict[u]
                    continue
                while used_point < len(used_g) and next_g == used_g[used_point]:
                    next_g += 1
                    used_point += 1
                replace[i] = next_g
                next_g += 1
            print(replace)
            l.nets = replace[numpy.searchsorted(unique, r[0])]

        if recalc:
            g = networkx.Graph()
            combine = numpy.arange(max(used_g+[next_g])+1)
            layers = {l.name: l for l in self.layers}
            for x in self.project["pins"]:
                r = layers[x[4]].nets[x[1], x[0]]
                for l in x[5:]:
                    g.add_edge(r, layers[l].nets[x[1], x[0]])
            for conn in networkx.connected_components(g):
                conn = list(conn)
                ex = [x for x in conn if x in self.project["nets"]]
                print(conn, ex)
                if ex:
                    tg = ex[0]
                else: tg = conn[0]
                for x in conn:
                    combine[x] = tg
            for l in self.layers:
                l.nets = combine[l.nets]

    def generate_netlist_v3(self, recalc=True, genfile=True):
        layers = {l.name: l for l in self.layers}
        pins_mapped = [layers[p[4]].nets[p[1], p[0]] for p in self.project["pins"]]

        total_nets = 2
        for l in self.layers:
            if l.name in ["__component__", "__boundary__"] or not l.mode&layer.Layer.NETLIST: continue
            l.nets, n = find_loops.find_traces_fast(l.layer[:, :, 0], True, total_nets)
            total_nets += n

        unused = [x for x in range(2, total_nets) if x not in pins_mapped]

        g = networkx.Graph()
        for x in self.project["pins"]:
            r = layers[x[4]].nets[x[1], x[0]]
            for l in x[5:]:
                g.add_edge(r, layers[l].nets[x[1], x[0]])
                print("adding edge", r, layers[l].nets[x[1], x[0]], x)

        conn = [list(x) for x in networkx.algorithms.connected_components(g)]
        print("connected components", conn)
        local_to_group = {l: i for i, v in enumerate(conn) for l in v}
        local_to_global = {}
        global_to_local = {}
        print(local_to_group)
        eliminated_old = []
        for p, old in zip(self.project["pins"], pins_mapped):
            new = layers[p[4]].nets[p[1], p[0]]
            if new <= 1 or old <= 1: continue
            new_gr = conn[local_to_group[new]][0] if new in local_to_group else new
            if new_gr in local_to_global and local_to_global[new_gr] != old and old not in global_to_local:
                print("Merged two nets", old, local_to_global[new_gr], "into", new_gr)
                if old in self.project["nets"]:
                    print("   Old net", old, "has priority")
                    eliminated_old.append(local_to_global[new_gr])
                    del global_to_local[local_to_global[new_gr]]
                    local_to_global[new_gr] = old
                    global_to_local[old] = new_gr
                else: eliminated_old.append(old)
            if old in global_to_local and global_to_local[old] != new_gr and new_gr not in local_to_global:
                new_global = unused.pop(0)
                print("Split net", old, "into", new_gr, global_to_local[old], "creating new net", new_global)
                local_to_global[new_gr] = new_global
                global_to_local[new_global] = new_gr
            if new_gr not in local_to_global and old not in global_to_local:
                local_to_global[new_gr] = old
                global_to_local[old] = new_gr

        print(local_to_global)
        print(global_to_local)
        print(eliminated_old)

        replace = numpy.arange(total_nets, dtype=numpy.uint32)
        for c in conn:
            for i in c: replace[i] = c[0]
        print(unused)
        # Assign net numbers to nets with no pins on them
        # TODO: make sure that new nets in the same group have the same number
        for i in range(2, total_nets):
            if replace[i] in local_to_global:
                replace[i] = local_to_global[replace[i]]
            else:
                val = unused.pop(0)
                local_to_global[replace[i]] = val
                replace[i] = val
        for l in self.layers:
            if l.name in ["__component__", "__boundary__"] or not l.mode&layer.Layer.NETLIST: continue
            l.nets = replace[l.nets]


    def export_netlist(self, use="spice"):
        net_cls = {
            "spice": netlist.SpiceNetlistExporter,
            "gschem": netlist.GschemExporter,
            "pcb": netlist.PCBNetlistExporter
        }
        net = net_cls[use](self.project["pins"], self.layers, self.project["nets"], self.project["components"])
        fname = tkinter.filedialog.asksaveasfilename()
        net.save_to_file(fname or "/dev/stdout", self.pname)

    def generate_netlist_v1(self, recalc=True, genfile=True):
        # TODO: Add a pinseq attribute for the pins.
        #   Possibly use a dict rather than a list to store the pin attributes. Or insert element before layer list

        # Algorithm:
        #   For every layer:
        #       Map each pixel to a number that uniquely (for that layer) identifies the net that pixel is on
        #       Replace these newly calculated numbers with the previous ones, matching whenever possible
        #   Count off all of the nets so that every net has a globally unique number.
        #   Construct a dict mapping from the globally unique number to a netname, when applicable
        #   Construct a list of globally unique numbers that refer to named nets.
        #   Call merge_loops on the set of pins connecting layers. Let size be the total number of nets
        #   Execute the desired merges
        #   Construct the netlist file, substituting a name for a number when appropriate

        # Note:
        #   Only up to 65536 nets total across all layers.

        t = time.time()
        layers = {l.name: l for l in self.layers}
        print("GENERATING NETLIST", recalc, genfile)
        for l in self.layers:
            if l.name in ["__component__", "__boundary__"] or not l.mode&layer.Layer.NETLIST: continue
            # mapped maps from layer-specific net numbers to globally unique net numbers
            l.mapped = numpy.zeros(65536, dtype=numpy.uint16)
            if not recalc: continue
            if l.nets_changed or True:
                print("Recomputing layer {}".format(l.name))
                #r = find_loops.find_traces(l.layer[:, :, 0]/255, prev=None, use_pb=True)
                r = find_loops.find_traces_fast(l.layer[:, :, 0]/255, use_pb=True)
                # Map between old and new nets
                p = []
                # rename maps from computed nets to the previous net: rename[computed] = old
                # available maps from old to new
                rename = numpy.zeros(65536, dtype=numpy.uint16)
                available = numpy.arange(0, 65536, dtype=numpy.uint16)
                for x in self.project["pins"]:
                    if l.name not in x[4:]: continue
                    if l.nets[x[1], x[0]] == 0: continue
                    print("{}/{} used to be on {}, now on {}".format(x[2], x[3], l.nets[x[1], x[0]], r[0][x[1], x[0]]))
                    p.append(x[:4] + [l.nets[x[1], x[0]]])
                    if available[l.nets[x[1], x[0]]]: rename[r[0][x[1], x[0]]] = l.nets[x[1], x[0]]
                    available[l.nets[x[1], x[0]]] = 0
                i = 1
                for x in available:
                    if x == 0: continue
                    while i < 65536 and rename[i] != 0: i+=1
                    rename[i] = x

                print("rename for {}:".format(l.name), rename[:20], available[:20])

                l.nets = rename[r[0]]
                l.nets_changed = False


        n = 1
        to_merge = []
        for x in self.project["pins"]:
            m = []
            for l in x[4:]:
                if layers[l].mode & layer.Layer.NETLIST == 0: continue
                nn = layers[l].nets[x[1], x[0]]
                if layers[l].mapped[nn] == 0:
                    layers[l].mapped[nn] = n
                    n += 1
                m.append(layers[l].mapped[nn])
            to_merge.append(m)
        # starts[globally unique] = user-defined netname
        starts = {}
        for x in self.project["nets"]:
            y = x.split(":", 1)
            nn = layers[y[1]].mapped[int(y[0])]
            layers[y[1]].rmap[nn] = int(y[0])
            new = self.project["nets"][x]
            starts[nn] = "0" if new == "GND" else new
        print("{} nets across all layers".format(n))
        if n == 1:
            print("No nets found")
            return
        print("to merge", to_merge)
        merged = find_loops.merge_loops(to_merge)
        print(merged, starts.keys())
        repl, fixed = find_loops.loops_to_replacements(merged, starts.keys())
        for l in self.layers:
            if l.name.startswith("__") or not l.mode&layer.Layer.NETLIST: continue
            print("mapping for", l.name, numpy.max(l.nets), "is", l.mapped[:numpy.max(l.nets)])
            print("rmap is", l.rmap[:200])
            l.mapped = repl[l.mapped]
            # Compute the inverse of l.mapped and store it to l.rmap
            # Since l.mapped is many to one, l.rmap points to the first one encountered
            for i in range(65536):
                v = l.mapped[i]
                l.rmap[v] = l.rmap[v] or i

        if not genfile: return
        connected = {}
        for x in self.project["pins"]:
            if not x[2]: continue
            l = layers[x[4]]
            if x[2] not in connected: connected[x[2]] = {}
            nn = l.mapped[l.nets[x[1], x[0]]]
            connected[x[2]][x[3]] = starts.get(nn, str(nn))
        """
        # Note on the replacement table:
        #   The key of the replacement table must be absolute (of the form number:layer)
        #   The value may either be absolute or named (#netname)
        replace = {}
        cnets = {x: [] for x in replace}
        for x in self.project["pins"]:
            cl = [l for l in self.layers if l.name in x[4:]]
            if len(cl) == 1:
                ln = cl[0].name
                cl = cl[0].nets
                if cl[x[1], x[0]] == 0:
                    print("pin {} is not on a net!".format(x))
                else:
                    if x[2] not in connected: connected[x[2]] = {}
                    connected[x[2]][x[3]] = str(cl[x[1], x[0]]) + ":" + ln
            else:
                print("pin {} is connected to more than one layer".format(x))
                cl = [(l, "{}:{}".format(l.nets[x[1], x[0]], l.name)) for l in self.layers if l.name in x[4:]]
                print(cl)
                for l in cl:
                    for i in cl:
                        #if i != l and i[1] in cnets: cnets[i[1]].append(l[1])
                        if i != l: cnets[i[1]] = cnets.get(i[1], []) + [l]
                    if l[1] in replace and replace[l[1]].startswith("#"):
                        # If there is a named net, replace all connected nets with the named one
                        for il in cl:
                            replace[il[1]] = replace[l[1]]
                        break
                else:
                    # The pin is not connected to any named nets. Check if there are any unnamed nets
                    for l in cl:
                        if l[1] in replace:
                            for il in cl:
                                replace[il[1]] = replace[l[1]]
                            break
                    else:
                        # None of the nets that the pin is connected to are listed in replace.
                        # Pick the first net the pin is connected to and replace the rest
                        for il in cl:
                            replace[il[1]] = cl[0][1]
        print("replace", replace)
        print(" values", list(set(replace.values())))
        print("cnets", cnets)
        print("Generating connected array", connected)
        print(self.project["pins"])
        for x in self.project["pins"]:
            if len(x) <= 5: continue
            cl = [l for l in self.layers if l.name == x[4]]
            if not cl or x[2] not in connected: continue
            nn = "{}:{}".format(cl[0].nets[x[1], x[0]], x[4])
            connected[x[2]][x[3]] = replace.get(nn, nn)"""

        print(connected)
        unnamed_list = [None]
        new_name = tkinter.filedialog.asksaveasfilename()
        with open(new_name or "/dev/stdout", "w") as circ:
            circ.write(self.pname+"\n")
            for x in connected:
                pins = connected[x]
                pn = [y.lower() for y in pins]
                if "b" in pn and "c" in pn and "e" in pn:
                    # BJT found
                    pins["1"] = pins.pop("c", pins.pop("C"))
                    pins["2"] = pins.pop("b", pins.pop("B"))
                    pins["3"] = pins.pop("e", pins.pop("E"))
                if "d" in pn and "g" in pn and "s" in pn:
                    # MOSFET found
                    pins["1"] = pins.pop("d", pins.pop("D"))
                    pins["2"] = pins.pop("g", pins.pop("G"))
                    pins["3"] = pins.pop("s", pins.pop("S"))
                    pins["4"] = pins.pop("b", pins.pop("B", pins["3"]))   # If no body pin, use the source pin
                if "a" in pn and "k" in pn:
                    # Diode found
                    pins["1"] = pins.pop("a", pins.pop("A"))
                    pins["2"] = pins.pop("k", pins.pop("K"))
                circ.write(x + " " + " ".join(y[1] for y in sorted(pins.items(), key=lambda x: int(x[0]))))
                if x in self.project["components"]:
                    comp = self.project["components"][x]
                    val = [y[1] for y in comp if y[0] == "value"]
                    if val: circ.write(" " + str(val[0]))
                circ.write("\n")
            circ.write(".END\n")

    def add_pin(self, pins, components):
        for pin in pins:
            self.project["pins"].append(pin)
            self.component_layer.click(pin[0], pin[1], 1)

        for refdes in components:
            self.project["components"][refdes] = components[refdes]
        self.canvas.update_image()

    def update_all_pins(self):
        print("Updating pins")
        self.component_layer.layer = numpy.zeros(self.component_layer.layer.shape, dtype=numpy.uint8)
        for x in self.project["pins"]:
            self.component_layer.click(x[0], x[1], 1)
        self.canvas.update_image()

    def view_components(self):
        view_comp.ComponentViewDialog(self.project["pins"], self.project["components"])

    def find_net(self):
        self.selected_net = simpledialog.askstring(title="Net number", prompt="Net number")
        print(self.selected_net, self.project["nets"])
        if self.selected_net in self.project["nets"]:
            self.selected_net = self.project["nets"][self.selected_net]
        elif self.selected_net.isnumeric(): self.selected_net = int(self.selected_net)
        else:
            self.selected_net = None
        if self.selected_net is not None:
            t = time.time()
            for l in self.layers:
                if l.mode & layer.Layer.NETLIST:
                    l.dim_mask = numpy.tensordot(l.nets == self.selected_net, numpy.uint8([1, 1, 1]), axes=0)
            print("Recalculation took", time.time() - t)
        self.canvas.update_image()

    def check_circuit(self):
        layers = {l.name: l for l in self.layers}
        problem = False
        for x in self.project["pins"]:
            for l in x[4:]:
                if layers[l].nets[x[1], x[0]] == 0:
                    problem = True
                    print("{} does not connect to a net on layer {}".format(x, l))
        if not problem: print("No net issues found")

    # Context menu
    def show_context_menu(self, event):
        print("Opening menu", event)
        self.contextmenu.delete(0, tkinter.END)
        l = self.get_active_layer()
        self.contextmenu.add_command(label=("Hide" if l.mode&layer.Layer.VISIBLE else "Show"), command=self.hide_show_layer)
        if l.name == "__component__":
            self.contextmenu.add_command(label="Refresh", command=self.update_all_pins)
        elif l.name != "__boundary__":
            self.contextmenu.add_command(label=("Don't netlist" if l.mode & layer.Layer.NETLIST else "Do netlist"),
                                         command=lambda: l.__setattr__("mode", l.mode ^ layer.Layer.NETLIST))
        try:
            self.contextmenu.tk_popup(event.x_root, event.y_root)
        finally:
            self.contextmenu.grab_release()

    # Functions to update drawing settings
    def update_opacity(self, evt):
        new = int(self.opacity_entry.get())/256
        #if new == self.opacity or new < 0 or new > 1: return
        self.opacity_scale.set(int(self.opacity_entry.get()))
        #self.opacity = new
        #self.on_scroll()

    def update_scale(self, event):
        new = int(self.opacity_scale.get())/256
        print("opacity from scale is", new)
        if new == self.opacity: return
        self.opacity_entry.delete(0, "end")
        self.opacity_entry.insert(0, str(self.opacity_scale.get()))
        self.opacity = new
        self.canvas.update_image()

    def apply_thickness(self, event):
        t = int(self.layers_thick.get())
        self.get_active_layer().thickness = t

    def toggle_presets(self, event):
        p1 = int(self.opacity_p1.get())
        p2 = int(self.opacity_p2.get())
        if self.opacity == p1/256:
            self.opacity_scale.set(p2)
        else:
            self.opacity_scale.set(p1)
        self.update_scale(None)
        return "break"

    def set_color(self):
        print("picking layer color")
        selected = self.get_active_layer()
        selected.color = list(map(int, tkinter.colorchooser.askcolor(self._arr2color(selected.color))[0]))[::-1]
        self.layers_color.config(bg=self._arr2color(selected.color))

    def update_color(self):
        selected = self.get_active_layer()
        print("Setting for layer:", selected.color, selected.thickness)
        self.layers_color.config(bg=self._arr2color(selected.color))
        self.layers_thick.delete(0, tkinter.END)
        self.layers_thick.insert(0, str(selected.thickness))

    # Event handlers for layers
    def new_layer(self):
        self.layers.append(layer.Layer(tkinter.simpledialog.askstring("New layer", "Enter the name of the new layer"), shape=self.solder_arr.shape))
        self.layers_list.insert(tkinter.END, self.layers[-1].name)

    def export_layer(self):
        l = self.get_active_layer()
        img = l.layer * l.color
        fname = tkinter.filedialog.asksaveasfilename()
        if fname and fname is not None:
            cv2.imwrite(fname, img)

    def propagate_click(self, event):
        ex, ey = self.canvas.canvas_to_image(event.x_root-self.canvas.winfo_rootx(), event.y_root-self.canvas.winfo_rooty())
        l = self.get_active_layer()
        m = self.draw_mode.get()
        if m in [1, 2] and l.mode & layer.Layer.VISIBLE:
            l.click(int(ex), int(ey), m)
            self.canvas.update_image()
        elif m == 3 and l.name != "__component__":
            print("Click state is", event.state)
            # Create a dialog for entering the refdes and pin number
            view_comp.PinCreateDialog(self.project["pins"], self.project["components"], self.get_active_layer().name, int(ex), int(ey),
                                      event.state & 1, self.add_pin)

    def propagate_unclick(self, event):
        for l in self.layers:
            l.unclick()

    def propagate_motion(self, event):
        ex, ey = self.canvas.canvas_to_image(event.x_root - self.canvas.winfo_rootx(),
                                             event.y_root - self.canvas.winfo_rooty())
        if 0 <= ex < self.solder_arr.shape[1] and 0 <= ey < self.solder_arr.shape[0]:
            ex, ey = int(ex), int(ey)
            if self.component_layer.layer[ey, ex].any():
                closest = min(self.project["pins"], key=lambda p: abs(p[0] - ex) + abs(p[1] - ey))
                self.pin_label.config(text="refdes: {}\npinnumber: {}".format(closest[2], closest[3]))
            else:
                self.pin_label.config(text="refdes:\npinnumber:")

            if self.project["version"] == "2":
                for l in self.layers:
                    nn = l.nets[int(ey), int(ex)]
                    if nn > 1 and l.mode & 0b0100:
                        #print(self.project["nets"])
                        name = self.project["nets"].get(nn, nn)
                        self.net_label.config(text="netname: {}".format(name))
                        break
                else:
                    self.net_label.config(text="netname:")
            else:
                nn = 0
                l = None
                for _l in self.layers:
                    nn = _l.nets[int(ey), int(ex)]
                    if nn and _l.mode & 0b0100:
                        l = _l
                        break
                if l and l.mapped[nn]:
                    print("found net {} in layer {}, maps to {}".format(nn, l.name, l.mapped[nn]))
                    m = l.mapped[nn]
                    # print(m)
                    # m is always accurate (assuming nets have been properly connected)
                    # m is the globally unique net number for the net the user is hovering on.
                    # However, it may be necessary to look for a layer L such that L.rmap[m]!=0
                    for ll in self.layers:
                        if ll.mode and ll.rmap[m]:
                            ll_a = str(ll.rmap[m]) + ":" + ll.name
                            print(m, "->", ll_a, self.project["nets"].get(ll_a, m), ll.mapped[ll.rmap[m]])
                            self.net_label.config(text="netname: {}\nx: {}\ny: {}".format(self.project["nets"].get(ll_a, m), ex, ey))
                            break
                    else:
                        self.net_label.config(text="netname:\n\n")
                else:
                    self.net_label.config(text="netname:\n\n")

        if self.draw_mode.get() not in [1, 2]: return
        sel = self.get_active_layer()
        if sel.name == "__component__": return
        if sel.motion(int(ex), int(ey)):
            self.canvas.update_image()

    def hide_show_layer(self):
        l = self.get_active_layer()
        l.mode ^= 0b00000100
        self.canvas.update_image()

    def hflip_t(self, evt):
        self.solder_arr = numpy.array(self.solder_arr[:, ::-1, :])
        self.comp_arr = numpy.array(self.comp_arr[:, ::-1, :])
        for l in self.layers:
            l.layer = numpy.array(l.layer[:, ::-1, :])
            l.nets = numpy.array(l.nets[:, ::-1])
        w = self.solder_arr.shape[1]
        for p in self.project["pins"]:
            print(p, "becomes ", end="")
            p[0] = w - 1 - p[0]
            print(p)
        self.canvas.update_image()

    def vflip_t(self, evt):
        self.solder_arr = numpy.array(self.solder_arr[::-1, :, :])
        self.comp_arr = numpy.array(self.comp_arr[::-1, :, :])
        for l in self.layers:
            l.layer = numpy.array(l.layer[::-1, :, :])
            l.nets = numpy.array(l.nets[::-1, :])
        h = self.solder_arr.shape[0]
        for p in self.project["pins"]:
            print(p, "becomes ", end="")
            p[1] = h - 1 - p[1]
            print(p)
        self.canvas.update_image()

    def update_component(self, oldpin, attr, pin):
        # The user clicked on a pin but both the pin and the corresponding component can be edited.
        # oldpin[2] represents the refdes the pin had when the menu was opened. This also represents the
        # current name of the component that was connected to the pin when the menu was opened. This means
        # that if the used did not change any value corresponding to the component, but they reassigned
        # the pin, then the same component will be changed.
        if "refdes" in [x[0] for x in attr]:
            if oldpin[2] in self.project["components"]: self.project["components"].pop(oldpin[2])
            self.project["components"][[x[1] for x in attr if x[0]=="refdes"][0]] = attr
            print("setting {} in components".format([x[1] for x in attr if x[0]=="refdes"][0]))
        if len(pin) >= 4:
            self.project["pins"][self.project["pins"].index(oldpin)] = oldpin[:2] + pin[2:]
        else:
            self.project["pins"].remove(oldpin)
            self.update_all_pins()

    def edit_component(self, event):
        ex, ey = self.canvas.canvas_to_image(event.x_root - self.canvas.winfo_rootx(),
                                             event.y_root - self.canvas.winfo_rooty())
        ex, ey = int(ex), int(ey)
        if self.component_layer.layer[ey, ex].any():
            closest = min(self.project["pins"], key=lambda p: abs(p[0]-ex)+abs(p[1]-ey))
            print("{}-{} clicked, id is {}".format(closest[2], closest[3], id(closest)), closest, ex, ey)
            view_comp.PinEditDialog(lambda a, b: self.update_component(closest, a, b), self.project["components"].get(closest[2], []), closest)
        else:
            print("no pin clicked")
            if 0 <= ex < self.solder_arr.shape[1] and 0 <= ey < self.solder_arr.shape[0]:
                if self.project["version"] == "2":
                    for l in self.layers:
                        nn = int(l.nets[int(ey), int(ex)])
                        if nn > 1 and l.mode & 0b0100:
                            old_name = self.project["nets"].get(nn, nn)
                            new_name = simpledialog.askstring(title="Edit netname", prompt="Enter netname",
                                                              initialvalue=old_name)
                            if new_name: self.project["nets"][nn] = new_name
                            elif new_name == "" and nn in self.project["nets"]: del self.project["nets"][nn]
                else:
                    l = None
                    nn = 0
                    for _l in self.layers:
                        nn = _l.nets[int(ey), int(ex)]
                        if nn:
                            l = _l
                            break
                    if l:
                        m = l.mapped[nn]
                        # m is always accurate (assuming nets have been properly connected)
                        # However, it may be necessary to look for a layer L such that L.rmap[m]!=0
                        for ll in self.layers:
                            if ll.rmap[m]:
                                ll_a = str(ll.rmap[m]) + ":" + ll.name
                                old_name = self.project["nets"].get(ll_a, ll_a)
                                new_name = simpledialog.askstring(title="Edit netname", prompt="Enter netname",
                                                                  initialvalue=old_name)
                                print(self.project["nets"])
                                if new_name: self.project["nets"][ll_a] = new_name
                                elif new_name == "" and ll_a in self.project["nets"]:
                                    del self.project["nets"][ll_a]
                                break

    def transform_dilate(self):
        l = self.get_active_layer()
        res = int(tkinter.simpledialog.askstring("Dilate", "Kernel size"))
        k = numpy.ones((res, res))
        l.layer = cv2.dilate(l.layer, k)
        self.update_images()

    def transform_erode(self):
        l = self.get_active_layer()
        res = int(tkinter.simpledialog.askstring("Erode", "Kernel size"))
        k = numpy.ones((res, res))
        l.layer = cv2.erode(l.layer, k)
        self.update_images()

    def transform_sobel(self):
        hsv = cv2.cvtColor(self.solder_arr, cv2.COLOR_BGR2HSV)
        l = cv2.cvtColor(self.solder_arr, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(l, cv2.CV_32F, 1, 0)
        grad_y = cv2.Sobel(l, cv2.CV_32F, 0, 1)
        grad = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
        cv2.imshow("grad", grad)
        cv2.imshow("hue", numpy.uint8((numpy.sin(numpy.pi*hsv[:, :, 0]/90)+1)*255))
        cv2.waitKey()

    def transform_fill_holes(self):
        l = self.get_active_layer().layer
        fill_holes.fill_holes(l[:, :, 0], thr_stdev=int(tkinter.simpledialog.askstring("Fill holes", "Max stdev")))
        l[:, :, 1] = l[:, :, 0]
        l[:, :, 2] = l[:, :, 0]

    def transform_block_average(self):
        l = self.get_active_layer().layer
        so = self.solder_arr
        s = numpy.int32(so)
        print(l.shape, s.shape, numpy.average(l, axis=(0,1)))
        for r in range(0, s.shape[0], 100):
            for c in range(0, s.shape[1], 100):
                s[r:r+100, c:c+100] = numpy.average(s[r:r+100, c:c+100][l[r:r+100,c:c+100, 0]!=0], axis=0)
        radius = 5

        def clicked(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                x = (x//100)*100
                y = (y//100)*100
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1, projection="3d")
                v = self.solder_arr[y:y+100, x:x+100].reshape((-1, 3))[:, ::-1]
                l = self.get_active_layer().layer[y:y+100, x:x+100].reshape((-1, 3))
                ax.scatter(*(v[l[:, 0] == 0]).transpose(), c=(v[l[:, 0] == 0])/255, marker="1")
                ax.scatter(*(v[l[:, 0] != 0]).transpose(), c="black", marker="o")
                cv2.imwrite("/tmp/tile.png", self.solder_arr[y:y+100, x:x+100])
                cv2.imwrite("/tmp/tile_layer.png", self.get_active_layer().layer[y:y+100, x:x+100]*255)
                plt.show()

        cv2.imshow("avg", numpy.uint8(s))
        cv2.setMouseCallback("avg", clicked)
        cv2.waitKey()
        """dist = (so[:, :, 0]-s[:, :, 0])**2 + (so[:, :, 1]-s[:, :, 1])**2 + (so[:, :, 2]-s[:, :, 2])**2
        so = cv2.cvtColor(cv2.cvtColor(so, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        so = cv2.cvtColor(numpy.uint8(2550 * dist / numpy.max(dist)), cv2.COLOR_GRAY2BGR)
        k = 0
        while k != ord("q"):
            if k == ord("w"): radius += 1
            if k == ord("s"): radius -= 1
            print("radius is", radius)
            in_d = numpy.tensordot(numpy.where(dist <= radius*radius, 1, 0), [0, 0, 255], 0)
            sq = cv2.addWeighted(so, 0.5, l*numpy.uint8([255, 0, 0]) | numpy.uint8(in_d), 0.5, 1)
            cv2.imshow("block avg", sq)
            k = cv2.waitKey()"""

    def on_components_identified(self, points):
        print("Identified", points)


if __name__ == "__main__":
    numpy.set_printoptions(threshold=numpy.inf)
    mw = MainWindow()
    mw.open_project("/Users/matthias/Documents/reverse_engineering/inverter.zip")
    #mw.open_project("/Users/matthias/Documents/Python/EdgeDetect/net_test_v2.zip")
    mw.mainloop()
