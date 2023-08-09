from cv2 import cv2
import numpy


def find_runs_fast(img):
    res = []
    for row in img:
        start = numpy.where(row[1:] > row[:-1])[0]+1
        stop = numpy.where(row[1:] < row[:-1])[0]+1
        runs = []
        for a, b in zip(stop[:-1], start[1:]):
            runs.append([a, b, 0])
        res.append(runs)
    return res


def fill_holes(img, thr_stdev=float("inf"), thr_area=float("inf")):
    holes = {0: 0, 1: 1}
    starts = numpy.argmax(img, axis=1)
    ends = img.shape[1] - numpy.argmax(img[:, ::-1], axis=1)
    print(starts)
    print(ends)
    all_runs = find_runs_fast(img)
    curr = 2
    for rn, row in enumerate(img):
        if not all_runs[rn] or rn == 0 or rn == img.shape[0]-1: continue
        if img[rn-1][starts[rn-1]] == 0: continue
        for n, (s, e, v) in enumerate(all_runs[rn]):
            # Check if the hole is open
            if s < starts[rn-1]: continue
            if e > ends[rn-1]: continue
            # Check if the run is new
            if numpy.all(img[rn-1, s:e]) and not (s < starts[rn+1] or e > ends[rn+1]):
                all_runs[rn][n][2] = curr
                holes[curr] = curr
                curr += 1
            else:
                # Otherwise, determine which runs we intersect
                rel = [r for r in all_runs[rn-1] if r[0] < e and r[1] > s]
                sn = set(holes[r[2]] for r in rel)
                if s < starts[rn+1] or e > ends[rn+1] or len(sn) < len(rel) or min(sn) == 0:
                    # If duplicate is found or the hole is open, invalidate the holes
                    for r in rel: holes[r[2]] = 0
                else:
                    for r in rel: holes[r[2]] = min(sn)
                    all_runs[rn][n][2] = min(sn)
    areas = {}
    for rn, row in enumerate(all_runs):
        for s, e, v in row:
            if not v or v not in holes: continue
            h = v
            while holes[h] != h: h = holes[h]
            hn = -1
            while hn != v: hn, holes[v], v = v, h, holes[v]
            if h:
                if h not in areas: areas[h] = [0, 0, 0, 0, 0]
                areas[h][0] += e - s
                areas[h][1] += (s + e - 1)*(e - s)//2
                areas[h][2] += (e - s)*rn
                areas[h][3] += (e - s)*(2*s*s + 2*s*e + 2*e*e - 3*e - 3*s + 1)//6
                areas[h][4] += (e - s)*rn**2
    stdev_map = {x: numpy.sqrt((v[3]+v[4] - (v[1]**2+v[2]**2)/v[0])/v[0]) for x, v in areas.items()}
    allowed = {x for x in areas if stdev_map[x] < thr_stdev and areas[x][0] < thr_area}
    for rn, row in enumerate(all_runs):
        for s, e, v in row:
            if holes[v] in allowed: img[rn, s:e] = 1


if __name__ == "__main__":
    #img = cv2.imread("/tmp/top_holes.png")[50:650, 250:750]
    #img = cv2.imread("/tmp/holes.png")
    #img = cv2.imread("output/loop1.png")
    img = numpy.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=numpy.uint8)
    cv2.imshow("img", img)
    cv2.waitKey()
    #l = numpy.where(img.sum(axis=2), 1, 0)
    l = img
    fill_holes(l, thr_stdev=15)
    print(id(l))
    print(l)
    img = numpy.tensordot(l, [0, 255, 255], axes=0)
    cv2.imshow("img", numpy.uint8(img))
    cv2.waitKey()
