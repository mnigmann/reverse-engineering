import sys


def parse_spice_netlist(file, reverse=False):
    pins = {}
    nets = {}
    with open(file) as f:
        f.readline()
        for x in f:
            if x[0] == "*": continue
            x = x.strip().split(" <")
            x = x[0].split(" ")[:-1] if len(x) == 1 else x[0].split(" ")
            if x and x[0][0] == "U": print(x)
            if reverse and x and x[0][0] == "U" and len(x) % 2 == 1:
                # Chip with an even number of pins
                print(len(x[-(len(x)//2):]), len(x[:len(x)//2:-1]), x, end=" -> ")
                x[-(len(x)//2):] = x[:len(x)//2:-1]
                print(x)
            for i in range(1, len(x)):
                pins[(x[0], i)] = x[i]
                if x[i] not in nets: nets[x[i]] = []
                nets[x[i]].append((x[0], i))
    return pins, nets


def parse_pcb_netlist(file):
    pins = {}
    nets = {}
    with open(file) as f:
        for l in f:
            if "\t" in l:
                n, p = l.split("\t")
            else:
                p = l
            p = [tuple(pin.split("-")) for pin in p.strip().split(" ")]
            for pin in p:
                pins[pin] = n
            nets[n] = p
    return pins, nets


#pins1, nets1 = parse_spice_netlist("/tmp/net3", reverse=True)
#pins2, nets2 = parse_spice_netlist("/tmp/net2")
pins1, nets1 = parse_pcb_netlist("/tmp/net_export")
pins2, nets2 = parse_pcb_netlist("/tmp/net_sch")
print(pins1)
print(pins2)
nets1to2 = {}
nets2to1 = {}
for p1 in pins1:
    if p1 not in pins2: print(p1, "is not in net2")
    else:
        if pins1[p1] not in nets1to2: nets1to2[pins1[p1]] = set()
        nets1to2[pins1[p1]].add(pins2[p1])
for p2 in pins2:
    if p2 not in pins1: print(p2, "is not in net1")
    else:
        if pins2[p2] not in nets2to1: nets2to1[pins2[p2]] = set()
        nets2to1[pins2[p2]].add(pins1[p2])

for n, s in nets1to2.items():
    if len(s) == 1: continue
    print("Net {} in net1 (containing {}) connects nets {} in net2".format(n, nets1[n], s))
    intersect = None
    for x in s:
        if intersect is None: intersect = set(y[0] for y in nets2[x])
        else: intersect.intersection_update(y[0] for y in nets2[x])
        print("    {} -> {}".format(x, nets2[x]))
    print("    Intersection:", intersect)

for n, s in nets2to1.items():
    if len(s) == 1: continue
    print("Net {} in net2 (containing {}) connects nets {} in net1".format(n, nets2[n], s))
    intersect = None
    for x in s:
        if intersect is None: intersect = set(y[0] for y in nets1[x])
        else: intersect.intersection_update(y[0] for y in nets1[x])
        print("    {} -> {}".format(x, nets1[x]))
    print("    Intersection:", intersect)
