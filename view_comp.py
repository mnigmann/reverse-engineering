import tkinter
from tkinter import simpledialog


class ComponentViewDialog(tkinter.Toplevel):
    def __init__(self, pins, components):
        super().__init__()
        self.pins = pins
        self.components = components
        self.title("Components")


class PinEditDialog(tkinter.Toplevel):
    def __init__(self, command, comp, pin=None):
        super().__init__()
        self.command = command
        self.pin = pin
        self.title("Edit component")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.attr = tkinter.LabelFrame(self, text="Attributes", bg="white")
        self.attr.grid(row=0, column=0, sticky="news")
        self.attr.columnconfigure(1, weight=1)
        tkinter.Label(self.attr, text="Name").grid(row=0, column=0)
        tkinter.Label(self.attr, text="Value").grid(row=0, column=1)
        self.w_attr = []
        for i, x in enumerate(comp+[["", ""]]):
            n = tkinter.Entry(self.attr, width=15)
            n.insert(0, x[0])
            n.grid(row=i+1, column=0)
            n.row = i
            v = tkinter.Entry(self.attr, width=20)
            v.insert(0, x[1])
            v.grid(row=i+1, column=1, sticky="ew")
            v.row = i
            self.w_attr.append([n, v])

        if pin is not None:
            self.pin_attr = tkinter.LabelFrame(self, text="Pin attributes")
            self.pin_attr.grid(row=1, column=0, sticky="ew")
            self.pin_attr.columnconfigure(1, weight=1)
            tkinter.Label(self.pin_attr, text="refdes").grid(row=0, column=0)
            self.p_r = tkinter.Entry(self.pin_attr)
            self.p_r.insert(0, pin[2])
            self.p_r.grid(row=0, column=1, sticky="ew")
            tkinter.Label(self.pin_attr, text="pinnumber").grid(row=1, column=0)
            self.p_n = tkinter.Entry(self.pin_attr)
            self.p_n.insert(0, pin[3])
            self.p_n.grid(row=1, column=1, sticky="ew")
            tkinter.Label(self.pin_attr, text="Connecting layers:").grid(row=2, column=0, columnspan=2, sticky="ew")
            self.p_l = tkinter.Entry(self.pin_attr)
            self.p_l.insert(0, ",".join(pin[4:]))
            self.p_l.grid(row=3, column=0, columnspan=2, sticky="ew")

        tkinter.Button(self, text="Done", command=self.finish).grid(row=2, column=1, sticky="e")
        tkinter.Button(self, text="Delete pin", command=self.delete_pin).grid(row=2, column=0, sticky="w")
        self.bind_all("<KeyPress>", self.add_row)

    def add_row(self, event):
        l = len(self.w_attr)
        w = event.widget
        if hasattr(w, "row") and w.row == l-1:
            print("last one", w.row, l)
            n = tkinter.Entry(self.attr, width=15)
            n.grid(row=l+1, column=0)
            n.row = l
            v = tkinter.Entry(self.attr, width=20)
            v.grid(row=l+1, column=1)
            v.row = l
            self.w_attr.append([n, v])

    def delete_pin(self):
        attr = [[x[0].get(), x[1].get()] for x in self.w_attr if x[0].get()]
        self.command(attr, self.pin[0:2])
        self.destroy()

    def finish(self):
        attr = [[x[0].get(), x[1].get()] for x in self.w_attr if x[0].get()]
        self.command(attr, [self.pin[0], self.pin[1], self.p_r.get(), self.p_n.get()] + [nl.strip() for nl in self.p_l.get().split(",")])
        self.destroy()


class PinCreateDialog(tkinter.Toplevel):
    def __init__(self, pins, components, default_layers, ex, ey, shift, command, refdes="", pinnumber="", layers=""):
        super().__init__()

        self.pins = pins
        self.components = components
        self.refdes = refdes
        self.pinnumber = pinnumber
        self.layers = layers
        self.default_layers = default_layers
        self.ex, self.ey, self.shift = ex, ey, shift
        self.command = command
        print("shift", self.shift)

        self.columnconfigure(1, weight=1)
        tkinter.Label(self, text="refdes").grid(row=0, column=0, sticky="w")
        self.rvar = tkinter.StringVar()
        self.f = tkinter.Entry(self, textvariable=self.rvar)
        self.f.grid(row=0, column=1, sticky="we")
        self.f.bind("<FocusOut>", self.check_refdes)

        tkinter.Label(self, text="pinnumber").grid(row=1, column=0, sticky="w")
        self.s = tkinter.Entry(self)
        self.s.grid(row=1, column=1, sticky="we")

        tkinter.Label(self, text="Connect to layers:").grid(row=2, column=0, columnspan=2, sticky="ew")
        self.l = tkinter.Entry(self)
        self.l.grid(row=3, column=0, columnspan=2, sticky="ew")

        tkinter.Button(self, text="Add", command=self.finish).grid(row=4, column=0, columnspan=2, sticky="e")
        self.bind("<Return>", self.finish)
        self.update()
        if not refdes: self.f.focus_force()
        elif not pinnumber: self.s.focus_force()
        else: self.l.focus_force()

    def check_refdes(self, event):
        refdes = self.rvar.get()
        base = refdes.strip("?")
        digits = len(refdes) - len(base)
        if digits > 0:
            largest = 0
            for x in self.components:
                if x.startswith(base) and x[len(base):].isnumeric() and len(x) == len(refdes):
                    largest = max(largest, int(x[len(base):]))
            self.rvar.set(base+str(largest+1).rjust(digits, "0"))

    def finish(self, event=None):
        refdes = self.rvar.get()
        pinnumber = self.s.get()
        layers = self.l.get()
        self.destroy()
        print("Adding pin at", refdes, pinnumber, "to layers", layers)
        ll = self.default_layers + ([nl.strip() for nl in layers.split(",")] if layers.strip() else [])
        target = []
        target_comp = self.components
        if self.pins and self.shift:
            last = self.pins[-1]
            n = int(pinnumber) - int(last[3])
            sx = (self.ex - last[0]) / n
            sy = (self.ey - last[1]) / n
            for x in range(1, 1 + n):
                target.append([int(last[0] + sx * x), int(last[1] + sy * x), refdes, str(int(last[3]) + x)] + ll)
        else:
            target.append([int(self.ex), int(self.ey), refdes, pinnumber] + ll)

        if refdes not in self.components:
            target_comp[refdes] = [["refdes", refdes]]
        elif refdes[0] in "RC" and not any(x[0]=="value" for x in self.components[refdes]):
            res = simpledialog.askstring(title="Value", prompt="Resistance, press enter to skip")
            if res: target_comp[refdes] = [["refdes", refdes], ["value", res]]
        self.command(target, target_comp)


if __name__ == "__main__":
    tk = tkinter.Tk()
    p = PinEditDialog(print, [["value", "10k"]], [1000, 1000, "R3", "1"])
    for x in p.w_attr:
        x[0].configure(state="disable")
        x[1].configure(state="disable")
    tk.mainloop()