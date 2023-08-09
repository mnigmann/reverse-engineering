import time

import cv2
import numpy
import random
import tkinter
import tkinter.ttk

def find_connected_line(arr, rn, cn):
    width = len(arr[0])
    start = 0
    stop = width
    for x in range(cn + 1, width):
        if arr[rn][x] != 1:
            stop = x
            break
    for x in range(cn-1, -1, -1):
        if arr[rn][x] != 1:
            start = x+1
            break
    return start, stop


def find_connected_runs(arr, rn, run):
    """
    Find runs in row rn+1 that are touching the given run located in row rn
    :param arr: a 2D array of ones and zeros
    :param rn: the index of the rown containing run
    :param run: an tuple of the form (start, stop) representing the initial run
    :return: a list of runs
    """
    res = []
    x = max(0, run[0]-1)
    while x < min(len(arr[0])-1, run[1]+1):
        if arr[rn+1][x]:
            new_run = find_connected_line(arr, rn+1, x)
            x = new_run[1]
            res.append(new_run)
        else: x += 1
    return res


def find_loop(arr_gray):
    startrow = 0
    startcol = 0
    for r in range(arr_gray.shape[0]):
        for c in range(arr_gray.shape[1]):
            if arr_gray[r, c]:
                startrow = r
                startcol = c
                break
        else: continue
        break
    run = find_connected_line(arr_gray, startrow, startcol)
    start = time.time()
    find_loop_rec(arr_gray, [run], startrow, 1)
    print("Finished in", time.time() - start)


def find_all_runs(arr_gray):
    res = []
    width = arr_gray.shape[1]
    for r in arr_gray:
        res.append([])
        c = 0
        while c < arr_gray.shape[1]:
            if r[c] == 1:
                start = c
                while c < width and r[c] == 1: c += 1
                res[-1].append((start, c))
            else: c += 1
    return res


def find_all_runs_fast(arr_gray):
    res = []
    for r in arr_gray:
        start = numpy.where(r[1:] > r[:-1])[0]+1
        stop = numpy.where(r[1:] < r[:-1])[0]+1
        runs = []
        for a, b in zip(start, stop):
            runs.append((a, b))
        res.append(runs)
    return res


def find_loop_rec(arr_gray, runs, rn, dir):
    all_runs = find_all_runs_fast(arr_gray)
    # print("runs are", all_runs)
    while rn < arr_gray.shape[0]:
        if not all_runs[rn]:
            rn += 1
            continue

        # Figure out which cells must be filled
        # If a run of cells is surrounded on three sides with borders, fill it
        #print("current row runs are", all_runs[rn])
        for ir in range(len(all_runs[rn])-1):
            gap_s = all_runs[rn][ir][1]
            gap_e = all_runs[rn][ir+1][0]
            for rp in all_runs[rn-1]:
                if rp[0] <= gap_s <= gap_e <= rp[1]:
                    # Fill gap_s:gap_e
                    # arr[rn, gap_s:gap_e] = [255, 0, 0]
                    arr_gray[rn, gap_s:gap_e] = 2
                    break
            if 2 in arr_gray[rn-1, gap_s:gap_e]:
                if 0 in arr_gray[rn-1, gap_s:gap_e]:
                    print("Breaking at row", rn)
                    # return False
                else:
                    # arr[rn, gap_s:gap_e] = [255, 0, 0]
                    arr_gray[rn, gap_s:gap_e] = 2

        #for r in all_runs[rn]: arr[rn, slice(*r)] = [0, 0, 255]
        #cv2.imshow("rn = {}".format(rn), cv2.resize(arr, (arr.shape[1] * 10, arr.shape[0] * 10)))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #for r in all_runs[rn]: arr[rn, slice(*r)] = [255, 255, 255]

        rn += 1

    # Cleanup step:
    # Starting from bottom, clear any filled runs directly above empty runs
    while rn > 0:
        rn -= 1

        for ir in range(len(all_runs[rn]) - 1):
            gap_s = all_runs[rn][ir][1]
            gap_e = all_runs[rn][ir + 1][0]
            if 2 in arr_gray[rn, gap_s:gap_e] and 0 in arr_gray[rn+1, gap_s:gap_e]:
                arr_gray[rn, gap_s:gap_e] = 0
                # arr[rn, gap_s:gap_e] = [0, 0, 0]

    #cv2.imshow("clean image", arr)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


def print_res(res, num, scale=1):
    colors = numpy.zeros((65536, 3), dtype=numpy.uint8)
    for x in range(1, 65536):
        c = [random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)]
        colors[x] = c
    out = colors[res]
    out = cv2.resize(out, (out.shape[1]*scale, out.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("/tmp/colored.png", out)
    cv2.imshow("res", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_traces(arr_gray, use_pb=False, prev=None, n=0):
    res = numpy.zeros(arr_gray.shape, dtype=numpy.uint16)
    #col = numpy.zeros(arr_gray.shape+(3,), dtype=numpy.uint8)
    #col[arr_gray == 1] = [255, 255, 255]
    rn = 0
    runs = [numpy.array(x) for x in find_all_runs_fast(arr_gray)]
    #print(runs)
    if use_pb:
        tk = tkinter.Toplevel()
        tkinter.Label(tk, text="Processing image...").grid(row=0, column=0)
        pb = tkinter.ttk.Progressbar(tk, orient=tkinter.HORIZONTAL, length=200, mode="determinate")
        pb.grid(row=1, column=0)
        tk.update()
    while rn < len(arr_gray):
        if not arr_gray[rn].any():
            rn += 1
            continue

        for r in runs[rn]:
            if not arr_gray[rn-1, r[0]:r[1]].any():
                # print("New run found", rn, r)
                #col[rn, r[0]:r[1]] = [random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)]
                #print(col[rn, r[0]])
                n += 1
                res[rn, r[0]:r[1]] = n
            else:
                # Check if there are any runs in the previous row that are connected to this one
                intersected = 0
                #c_inter = [0, 0, 0]
                rel = runs[rn-1][numpy.array((runs[rn-1][:, 0] <= r[1]) & (runs[rn-1][:, 1] >= r[0]), dtype=bool)]
                for pr in rel:
                    # if r[0] <= pr[1] and r[1] >= pr[0]:
                        if not intersected:
                            #print("row {}: Run {} intersects previous {}".format(rn, r, pr))
                            #col[rn, r[0]:r[1]] = col[rn-1, pr[0]]
                            res[rn, r[0]:r[1]] = res[rn-1, pr[0]]
                            intersected = res[rn-1, pr[0]]
                            #c_inter = col[rn-1, pr[0]]
                        else:
                            #print("row {}: Run {} intersects additional run {}".format(rn, r, pr))
                            c = res[rn-1, pr[0]]
                            #col[res == c] = c_inter
                            if c != intersected: res[res == c] = intersected
        if use_pb and pb['value'] != int(rn/len(arr_gray) * 100):
            pb['value'] = int(rn / len(arr_gray) * 100)
            tk.update()

        #cv2.imshow("rn = {}".format(rn), cv2.resize(col, (500, 500), interpolation=cv2.INTER_NEAREST))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        rn += 1
    if use_pb:
        tk.destroy()
    return res, n


def find_traces_fast(arr_gray, use_pb=False, n=1):
    res = numpy.zeros(arr_gray.shape, dtype=numpy.uint16)
    rn = 0
    runs = [numpy.array(x) for x in find_all_runs_fast(arr_gray)]
    replace = numpy.zeros((65536,), dtype=numpy.uint32)
    if use_pb:
        tk = tkinter.Toplevel()
        tkinter.Label(tk, text="Processing image...").grid(row=0, column=0)
        pb = tkinter.ttk.Progressbar(tk, orient=tkinter.HORIZONTAL, length=200, mode="determinate")
        pb.grid(row=1, column=0)
        tk.update()
    while rn < len(arr_gray):
        if not arr_gray[rn].any():
            rn += 1
            continue
        #print(rn, runs[rn])
        for r in runs[rn]:
            if not arr_gray[rn-1, r[0]:r[1]].any():
                # A new run is found
                #print("    adding", rn, r, n)
                res[rn, r[0]:r[1]] = n
                replace[n] = n
                n += 1
            else:
                # Check if there are any runs in the previous row that are connected to this one
                intersected = 0
                #c_inter = [0, 0, 0]
                rel = runs[rn-1][numpy.array((runs[rn-1][:, 0] <= r[1]) & (runs[rn-1][:, 1] >= r[0]), dtype=bool)]
                #print("    {} {} {}".format(rn, r, rel))
                for pr in rel:
                    # if r[0] <= pr[1] and r[1] >= pr[0]:
                        if not intersected:
                            #print("    row {}: Run {} intersects previous {}".format(rn, r, pr), res[rn-1, pr[0]])
                            #col[rn, r[0]:r[1]] = col[rn-1, pr[0]]
                            res[rn, r[0]:r[1]] = res[rn-1, pr[0]]
                            intersected = res[rn-1, pr[0]]
                            #c_inter = col[rn-1, pr[0]]
                        else:
                            # This run connects two other runs. It is necessary to merge them.
                            c = res[rn-1, pr[0]]
                            if not are_connected(replace, intersected, c):
                                t = replace[c]
                                replace[c] = replace[intersected]
                                replace[intersected] = t
        if use_pb and pb['value'] != int(rn/len(arr_gray) * 100):
            pb['value'] = int(rn / len(arr_gray) * 100)
            tk.update()
        rn += 1
    replace = numpy.array(loops_to_replacements(replace, [])[0], dtype=numpy.uint16)
    print(replace.dtype, replace.shape)
    res = replace[res]
    if use_pb:
        tk.destroy()
    return res, n


def are_connected(connected, a, b):
    c_a = a
    for x in range(0, len(connected)):
        # If there is a problem and a is not in a loop, the program will break at some point
        n_a = connected[c_a]
        if n_a == a: return False
        if n_a == b: return True
        c_a = n_a


def merge_loops(pins):
    connected = numpy.arange(0, len(pins), dtype=numpy.uint16)
    for x in pins:
        src = x[0]
        for l in x[1:]:
            if connected[src] == src or connected[l] == l: pass
            if src == l or are_connected(connected, src, l): continue
            t = connected[src]
            connected[src] = connected[l]
            connected[l] = t
    return connected


def set_loop(loops, pos, result):
    result[pos] = pos
    p = loops[pos]
    x = 0
    while p != pos and x < len(loops):
        result[p] = pos
        q = p
        p = loops[p]
        loops[q] = 0
        x += 1
    loops[pos] = 0


def loops_to_replacements(loops, starts):
    result = numpy.array(loops, dtype=numpy.uint16)
    loops = loops.copy()
    fixed = []
    for s in starts:
        set_loop(loops, s, result)
        fixed.append(s)
    for x in loops:
        if x == 0: continue
        set_loop(loops, x, result)
        fixed.append(x)
    return result, fixed


def get_loops(connected):
    conn = connected.copy()
    i = 0
    loop = []
    loops = []
    while i < len(conn):
        if loop and i == loop[0]:
            conn[i] = -1
            loops.append(list(sorted(loop)))
            loop = []
        elif conn[i] == -1: pass
        elif conn[i] == i:
            conn[i] = -1
            loops.append([i])
        else:
            loop.append(i)
            v = conn[i]
            conn[i] = -1
            i = v
            continue
        i += 1
    return loops


if __name__ == '__main__':
    #lp = numpy.array([0, 2, 3, 4, 1, 5, 6, 7], dtype=numpy.uint16)
    #res = numpy.zeros(len(lp), dtype=numpy.uint8)
    #print(loops_to_replacements(lp, [5]))
    img = cv2.imread("/Users/matthias/Documents/LAYERBottom.png")
    s = time.time()
    r = find_traces_fast(img[:, :, 0]/255)
    print("done in", time.time()-s, r)
    print_res(*r)
    """loop_img = cv2.imread("output/test.png")/255
    find_loop(loop_img[:, :, 0])
    loop_img[:, :, 0] = numpy.where(loop_img[:, :, 0] == 2, 1, 0)
    loop_img[:, :, 1] = loop_img[:, :, 0]
    loop_img[:, :, 2] = loop_img[:, :, 0]
    cv2.imshow("final", loop_img*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""