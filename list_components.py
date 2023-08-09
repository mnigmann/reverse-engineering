import tkinter
from tkinter import ttk
from tkinter import simpledialog

class ComponentListDialog(tkinter.Toplevel):
    def __init__(self, components: dict, pins: dict, scrollto_command):
        super().__init__()
        self.components = components
        self.pins = pins
        self.scrollto_command = scrollto_command

        self.table = ttk.Treeview(self)
        self.table["columns"] = ("refdes", "pins", "value", "symbol", "partnum")
        self.table.column("#0", width=0, stretch=tkinter.NO)
        self.table.column("refdes", anchor=tkinter.CENTER, width=50)
        self.table.column("pins", anchor=tkinter.CENTER, width=50)
        self.table.column("value", anchor=tkinter.CENTER, width=50)
        self.table.column("symbol", anchor=tkinter.CENTER, width=100)
        self.table.column("partnum", anchor=tkinter.CENTER, width=100)

        self.table.heading("#0", text="", anchor=tkinter.CENTER)
        self.table.heading("refdes", text="refdes", anchor=tkinter.CENTER)
        self.table.heading("pins", text="# pins", anchor=tkinter.CENTER)
        self.table.heading("value", text="Value", anchor=tkinter.CENTER)
        self.table.heading("symbol", text="Symbol file", anchor=tkinter.CENTER)
        self.table.heading("partnum", text="Part number", anchor=tkinter.CENTER)

        self.by_refdes = {}
        for pin in self.pins:
            if pin[2] not in self.by_refdes: self.by_refdes[pin[2]] = []
            self.by_refdes[pin[2]].append(pin[:2])

        c_sorted = sorted(self.components.items(), key=lambda x: int(x[0].upper().strip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")))
        c_sorted = sorted(c_sorted, key=lambda x: x[0].strip("0123456789"))

        for i, (refdes, attr) in enumerate(c_sorted):
            value = ""
            symbol = ""
            partnum = ""
            for key, val in attr:
                if key == "value": value = val
                if key == "symbol": symbol = val
                if key == "partnum": partnum = val
            self.table.insert(parent="", index="end", iid=refdes, text="", values=(
                refdes, len(self.by_refdes[refdes]) if refdes in self.by_refdes else "", value, symbol, partnum))
        self.table.bind("<<TreeviewSelect>>", self.on_select)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.table.grid(row=0, column=0, sticky="news")
        self.last_component = None
        self.index = 0

        self.sym_button = tkinter.Button(self, text="Set symbol for ''", command=self.set_symbol)
        self.sym_button["state"] = tkinter.DISABLED
        self.sym_button.grid(row=1, column=0, sticky="w")

    def set_symbol(self):
        sym = simpledialog.askstring(title="Set symbol", prompt="Enter symbol file")
        if sym is not None:
            target = self.table.item(self.table.selection()[0])["values"][4]
            print(self.table.item(self.table.selection()[0]))
            for x in self.components:
                partnum = [t[1] for t in self.components[x] if t[0] == "partnum"]
                if partnum and partnum[0] == target:
                    oldval = self.table.item(x)["values"]
                    self.components[x] = [t for t in self.components[x] if t[0] != "symbol"] + [["symbol", sym]]
                    print(x, "matches, oldval", oldval)
                    oldval[3] = sym
                    self.table.item(x, values=oldval)
        self.update()
    
    def on_select(self, event):
        refdes = self.table.selection()[0]
        if self.last_component == refdes:
            self.index += 1
        else: self.index = 0
        pins = self.by_refdes.get(refdes, [])
        self.scrollto_command(pins[self.index % len(pins)])
        self.last_component = refdes

        partnum = [x[1] for x in self.components[refdes] if x[0] == "partnum"]
        if partnum:
            self.sym_button["text"] = "Set symbol for '{}'".format(partnum[0])
            self.sym_button["state"] = tkinter.NORMAL
            self.update()
        else:
            self.sym_button["text"] = "Set symbol for ''"
            self.sym_button["state"] = tkinter.DISABLED
            self.update()


if __name__ == '__main__':
    d = ComponentListDialog({"R1": [["refdes", "R1"], ["value", "33k"]], "Q1": [["refdes", "Q1"]]})
    tkinter.mainloop()