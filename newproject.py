import tkinter
from tkinter import filedialog
import cv2


class NewProjectDialog(tkinter.Toplevel):
    def __init__(self, on_complete):
        super().__init__()

        self.title("New project")
        self.on_complete = on_complete

        tkinter.Label(self, text="New filename").grid(row=0, column=0)
        self.new_name = tkinter.Entry(self)
        self.new_name.grid(row=0, column=1)
        self.comp_btn = tkinter.Button(self, text="Choose", command=lambda: self.filepicker(self.new_name, filedialog.asksaveasfilename))
        self.comp_btn.grid(row=0, column=2)

        tkinter.Label(self, text="Component side").grid(row=1, column=0)
        self.comp_entry = tkinter.Entry(self)
        self.comp_entry.grid(row=1, column=1)
        self.comp_btn = tkinter.Button(self, text="Choose", command=lambda: self.filepicker(self.comp_entry))
        self.comp_btn.grid(row=1, column=2)

        tkinter.Label(self, text="Solder side").grid(row=2, column=0)
        self.solder_entry = tkinter.Entry(self)
        self.solder_entry.grid(row=2, column=1)
        self.solder_btn = tkinter.Button(self, text="Choose", command=lambda: self.filepicker(self.solder_entry))
        self.solder_btn.grid(row=2, column=2)

        self.continue_btn = tkinter.Button(self, text="Continue", command=self.done)
        self.continue_btn.grid(row=3, column=2)

    def done(self):
        comp = self.comp_entry.get()
        solder = self.solder_entry.get()
        name = self.new_name.get()
        try:
            cv2.imread(comp)
            cv2.imread(solder)
        except:
            return
        try:
            open(name).close()
        except FileNotFoundError:
            self.destroy()
            self.on_complete(solder, comp, name)

    def filepicker(self, target, function=filedialog.askopenfilename):
        fname = function()
        target.delete(0, len(target.get()))
        target.insert(0, fname)

if __name__ == '__main__':
    n = NewProjectDialog(print)
    n.mainloop()
