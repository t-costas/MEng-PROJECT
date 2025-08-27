# panel_to_pcbs_with_step_dxf_yolo.py
# Same as your working script, plus YOLO label export for specific switch IDs:
# default switch ids: 2,4,6,8,10,12  (can override via --switch_ids)

import os, math, argparse, copy
import numpy as np
import cv2

# ----- Optional heavy deps (STEP/DXF) -----
try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib_Add
    HAVE_OCC = True
except Exception:
    HAVE_OCC = False

try:
    import ezdxf
    HAVE_EZDXF = True
except Exception:
    HAVE_EZDXF = False


# ---------- CLI ----------
p = argparse.ArgumentParser()
# Images
p.add_argument("--imageFilepath", required=True, help="Panel photo")
p.add_argument("--outputDirectory", default="./outputs_yolo")
p.add_argument("--pcb_out_size", type=int, default=900, help="Warped PCB size (square)")
# Fiducial template (pixels)
p.add_argument("--fiducialInnerDiameterInPixels", type=int, default=18)
p.add_argument("--fiducialOuterDiameterInPixels", type=int, default=48)
p.add_argument("--expected_fiducials", type=int, default=8)
# STEP & DXF
p.add_argument("--step_path", required=True, help="STEP file for component bboxes (mm)")
p.add_argument("--dxf_path", required=True, help="DXF file for board outline (mm)")
p.add_argument("--dxf_outline_layer", type=str, default=None,
               help="Layer name of board outline in DXF (e.g., 'Mechanical2', 'Edge.Cuts'). If None, uses all entities.")
p.add_argument("--dxf_y_up", action="store_true",
               help="Set if your DXF/STEP Y is up (invert Y when mapping to image which is Y-down).")
# Fiducials in mm (3 options)
p.add_argument("--ne_mm", type=str, default=None,
               help='(deprecated) NE fiducial as "x,y" in mm')
p.add_argument("--sw_mm", type=str, default=None,
               help='(deprecated) SW fiducial as "x,y" in mm')

p.add_argument("--ne_x_mm", type=float, default=None, help="NE fiducial X coordinate in mm (recommended)")
p.add_argument("--ne_y_mm", type=float, default=None, help="NE fiducial Y coordinate in mm (recommended)")
p.add_argument("--sw_x_mm", type=float, default=None, help="SW fiducial X coordinate in mm (recommended)")
p.add_argument("--sw_y_mm", type=float, default=None, help="SW fiducial Y coordinate in mm (recommended)")

p.add_argument("--fid_ne_id", type=int, default=None, help="Alternative: STEP solid id to use as NE fiducial")
p.add_argument("--fid_sw_id", type=int, default=None, help="Alternative: STEP solid id to use as SW fiducial")

# Crops for each component?
p.add_argument("--save_component_crops", action="store_true")

# YOLO export
p.add_argument("--export_yolo", action="store_true",
               help="Export YOLO labels from projected STEP bboxes (switch ids only).")
p.add_argument("--yolo_dir", type=str, default=None,
               help="Root folder for the YOLO dataset (default: <outputDirectory>/yolo).")
p.add_argument("--yolo_split", type=str, default="train",
               help="Split subfolder: train/val/test (default: train).")
p.add_argument("--yolo_class_names", type=str, default="switch",
               help="Comma-separated class names. Default single class 'switch'.")
p.add_argument("--switch_ids", type=str, default="2,4,6,8,10,12",
               help="Comma-separated STEP solid ids that are switches (defaults to '2,4,6,8,10,12').")

args = p.parse_args()
os.makedirs(args.outputDirectory, exist_ok=True)


# ---------- Helpers ----------
def _parse_id_list(s):
    if not s: return set()
    return set(int(x.strip()) for x in s.split(",") if x.strip() != "")

def _make_yolo_dirs(yolo_root, split):
    img_dir = os.path.join(yolo_root, "images", split)
    lab_dir = os.path.join(yolo_root, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    return img_dir, lab_dir

def _bbox_xyxy_to_yolo(x0, y0, x1, y1, img_w, img_h):
    # clamp to image & reject empties
    x0 = max(0, min(img_w - 1, x0))
    x1 = max(0, min(img_w - 1, x1))
    y0 = max(0, min(img_h - 1, y0))
    y1 = max(0, min(img_h - 1, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w  = (x1 - x0)
    h  = (y1 - y0)
    return (cx / img_w, cy / img_h, w / img_w, h / img_h)

def make_template(inner_d, outer_d):
    sz = outer_d if outer_d % 2 == 1 else outer_d + 1
    pat = np.zeros((sz, sz), dtype=np.uint8)
    cv2.circle(pat, (sz//2, sz//2), outer_d//2, 70, cv2.FILLED)
    cv2.circle(pat, (sz//2, sz//2), inner_d//2, 255, cv2.FILLED)
    pat = pat.astype(np.float32)
    pat = (pat - pat.mean()) / (pat.std() + 1e-6)
    return pat

def topk_peaks(match, k, suppress_radius):
    m = match.copy()
    peaks = []
    for _ in range(k):
        _, maxVal, _, maxLoc = cv2.minMaxLoc(m)
        if maxVal <= -1e-9:
            break
        y, x = maxLoc[1], maxLoc[0]
        peaks.append((y, x))
        y0 = max(0, y - suppress_radius); y1 = min(m.shape[0], y + suppress_radius + 1)
        x0 = max(0, x - suppress_radius); x1 = min(m.shape[1], x + suppress_radius + 1)
        m[y0:y1, x0:x1] = -1e-9
    return peaks

def detect_fiducials(gray, pat, expected):
    match = cv2.matchTemplate(gray.astype(np.float32), pat, cv2.TM_CCOEFF_NORMED)
    ph, pw = pat.shape
    peaks = topk_peaks(match, expected, suppress_radius=max(6, ph//2))
    centers = [(int(x + pw//2), int(y + ph//2)) for (y, x) in peaks]
    return centers

def angle_deg(vx, vy):
    a = math.degrees(math.atan2(vy, vx))
    if a < 0: a += 360.0
    return a

def all_candidate_pairs(points):
    n = len(points)
    raw, lens = [], []
    for i in range(n):
        xi, yi = points[i]
        for j in range(n):
            if i == j: continue
            xj, yj = points[j]
            if xi > xj and yi < yj:
                vx, vy = (xj - xi), (yj - yi)
                L = math.hypot(vx, vy)
                ang = angle_deg(vx, vy)
                raw.append((i, j, L, ang)); lens.append(L)
    if not raw: return []
    medL = np.median(lens)
    cands = []
    for (i, j, L, ang) in raw:
        ang_err = abs(ang - 135.0); ang_err = min(ang_err, 360 - ang_err)
        score = abs(L - medL) + 0.8 * ang_err
        cands.append((score, i, j, L, ang))
    cands.sort(key=lambda t: t[0])
    return cands

def choose_four_disjoint_pairs(points):
    cand = all_candidate_pairs(points)
    used, chosen = set(), []
    for score, i, j, L, ang in cand:
        if i in used or j in used: continue
        if abs(ang - 135.0) > 35.0: continue
        chosen.append((i, j))
        used.add(i); used.add(j)
        if len(chosen) == 4: break
    return chosen

def label_ne_sw(p_ne, p_sw):
    (x1, y1), (x2, y2) = p_ne, p_sw
    return (p_ne, p_sw) if (x1 > x2 and y1 < y2) else (p_sw, p_ne)

def load_step_components(step_path):
    if not HAVE_OCC:
        raise RuntimeError("pythonocc-core (OCC) not available in this interpreter.")
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    comps = []
    idx = 0
    while ex.More():
        solid = ex.Current()
        bbox = Bnd_Box()
        brepbndlib_Add(solid, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        comps.append({
            "id": idx,
            "bbox_mm": [float(xmin), float(ymin), float(zmin), float(xmax), float(ymax), float(zmax)]
        })
        idx += 1
        ex.Next()
    return comps

def pick_fiducials_mm(comps, ne_mm_arg, sw_mm_arg, ne_x, ne_y, sw_x, sw_y, fid_ne_id, fid_sw_id):
    if ne_mm_arg and sw_mm_arg:
        ne_x2, ne_y2 = [float(v) for v in ne_mm_arg.split(",")]
        sw_x2, sw_y2 = [float(v) for v in sw_mm_arg.split(",")]
        return (ne_x2, ne_y2), (sw_x2, sw_y2)
    if (ne_x is not None) and (ne_y is not None) and (sw_x is not None) and (sw_y is not None):
        return (float(ne_x), float(ne_y)), (float(sw_x), float(sw_y))
    if (fid_ne_id is not None) and (fid_sw_id is not None):
        ne_c = next(c for c in comps if c["id"] == fid_ne_id)
        sw_c = next(c for c in comps if c["id"] == fid_sw_id)
        ne_x3 = (ne_c["bbox_mm"][0] + ne_c["bbox_mm"][3]) / 2.0
        ne_y3 = (ne_c["bbox_mm"][1] + ne_c["bbox_mm"][4]) / 2.0
        sw_x3 = (sw_c["bbox_mm"][0] + sw_c["bbox_mm"][3]) / 2.0
        sw_y3 = (sw_c["bbox_mm"][1] + sw_c["bbox_mm"][4]) / 2.0
        return (ne_x3, ne_y3), (sw_x3, sw_y3)
    raise ValueError(
        "Provide fiducial mm using --ne_x_mm/--ne_y_mm/--sw_x_mm/--sw_y_mm, "
        "or --ne_mm/--sw_mm 'x,y', or --fid_ne_id/--fid_sw_id."
    )

def board_bbox_from_dxf(dxf_path, outline_layer=None):
    if not HAVE_EZDXF:
        raise RuntimeError("ezdxf not available in this interpreter.")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    def on_layer(e):
        return (outline_layer is None) or (getattr(e.dxf, "layer", None) == outline_layer)

    minx = miny = maxx = maxy = None
    def upd(x, y):
        nonlocal minx, miny, maxx, maxy
        minx = x if minx is None else min(minx, x)
        miny = y if miny is None else min(miny, y)
        maxx = x if maxx is None else max(maxx, x)
        maxy = y if maxy is None else max(maxy, y)

    for e in msp:
        if not on_layer(e): continue
        t = e.dxftype()
        try:
            if t == "LINE":
                s, en = e.dxf.start, e.dxf.end
                upd(s[0], s[1]); upd(en[0], en[1])
            elif t in ("LWPOLYLINE", "POLYLINE"):
                for pt in e.get_points("xy"): upd(pt[0], pt[1])
            elif t == "CIRCLE":
                c, r = e.dxf.center, e.dxf.radius
                upd(c[0]-r, c[1]-r); upd(c[0]+r, c[1]+r)
            elif t == "ARC":
                c, r = e.dxf.center, e.dxf.radius
                upd(c[0]-r, c[1]-r); upd(c[0]+r, c[1]+r)
            elif t == "SPLINE":
                for p in e.control_points: upd(p[0], p[1])
            elif t == "INSERT":
                ins = e.dxf.insert; upd(ins[0], ins[1])
        except Exception:
            continue

    if None in (minx, miny, maxx, maxy):
        raise RuntimeError("DXF bounds not found (try --dxf_outline_layer)")
    return float(minx), float(miny), float(maxx), float(maxy)

def board_rect_from_midpoint(ne_px, sw_px, NE_MM, SW_MM, minx_mm, miny_mm, maxx_mm, maxy_mm):
    ne_x_px, ne_y_px = ne_px
    sw_x_px, sw_y_px = sw_px

    dx_px = float(ne_x_px - sw_x_px)
    dy_px = float(sw_y_px - ne_y_px)
    dx_mm = float(NE_MM[0] - SW_MM[0])
    dy_mm = float(SW_MM[1] - NE_MM[1])
    if dx_mm == 0 or dy_mm == 0:
        raise ValueError("Bad fiducial mm: dx_mm or dy_mm is zero")

    sx = dx_px / dx_mm
    sy = dy_px / dy_mm

    W_mm = float(maxx_mm - minx_mm)
    H_mm = float(maxy_mm - miny_mm)

    cx_px = 0.5 * (ne_x_px + sw_x_px)
    cy_px = 0.5 * (ne_y_px + sw_y_px)

    hx_px = abs(sx) * (W_mm * 0.5)
    hy_px = abs(sy) * (H_mm * 0.5)

    left_x  = int(round(cx_px - hx_px))
    right_x = int(round(cx_px + hx_px))
    top_y   = int(round(cy_px - hy_px))
    bot_y   = int(round(cy_px + hy_px))

    return left_x, right_x, top_y, bot_y, sx, sy

def warp_axis_aligned_rect(bgr, left_x, right_x, top_y, bot_y, out_size):
    NW = (left_x,  top_y)
    NE = (right_x, top_y)
    SE = (right_x, bot_y)
    SW = (left_x,  bot_y)
    src = np.float32([NW, NE, SE, SW])
    dst = np.float32([[0,0], [out_size-1,0], [out_size-1,out_size-1], [0,out_size-1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(bgr, M, (out_size, out_size)), (NW, NE, SE, SW)

def to_px_mappers(out_size, minx_mm, miny_mm, maxx_mm, maxy_mm, y_up=False):
    W_mm = maxx_mm - minx_mm
    H_mm = maxy_mm - miny_mm
    def to_px_x(x_mm):
        return ((x_mm - minx_mm) / W_mm) * (out_size - 1)
    if not y_up:
        def to_px_y(y_mm):
            return ((y_mm - miny_mm) / H_mm) * (out_size - 1)
    else:
        def to_px_y(y_mm):
            return ((maxy_mm - y_mm) / H_mm) * (out_size - 1)
    return to_px_x, to_px_y

def flip_mm_180_about_board(x_mm, y_mm, minx_mm, miny_mm, maxx_mm, maxy_mm):
    cx = 0.5 * (minx_mm + maxx_mm)
    cy = 0.5 * (miny_mm + maxy_mm)
    return (2*cx - x_mm, 2*cy - y_mm)


# ---------- YOLO setup ----------
if args.export_yolo:
    yolo_root = args.yolo_dir if args.yolo_dir else os.path.join(args.outputDirectory, "yolo")
    YOLO_IMG_DIR, YOLO_LAB_DIR = _make_yolo_dirs(yolo_root, args.yolo_split)
    CLASS_NAMES = [c.strip() for c in args.yolo_class_names.split(",") if c.strip() != ""]
    SWITCH_IDS = _parse_id_list(args.switch_ids)
else:
    YOLO_IMG_DIR = YOLO_LAB_DIR = None
    CLASS_NAMES = ["switch"]
    SWITCH_IDS = set()


# ---------- Main ----------
def main():
    # Load image and detect fiducials
    bgr = cv2.imread(args.imageFilepath)
    if bgr is None:
        raise FileNotFoundError(args.imageFilepath)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    pat = make_template(args.fiducialInnerDiameterInPixels, args.fiducialOuterDiameterInPixels)
    centers = detect_fiducials(gray, pat, args.expected_fiducials)

    anno = copy.deepcopy(bgr)
    r = args.fiducialOuterDiameterInPixels // 2
    for (cx, cy) in centers:
        cv2.rectangle(anno, (cx - r, cy - r), (cx + r, cy + r), (0,255,0), 2)

    # Pair NE↘SW
    pairs = choose_four_disjoint_pairs(centers)
    if len(pairs) < 4:
        print(f"Warning: only formed {len(pairs)} pairs; some PCBs may be skipped.")

    # STEP comps + pick fiducial mm
    comps = load_step_components(args.step_path)
    NE_MM, SW_MM = pick_fiducials_mm(
        comps,
        args.ne_mm, args.sw_mm,
        args.ne_x_mm, args.ne_y_mm, args.sw_x_mm, args.sw_y_mm,
        args.fid_ne_id, args.fid_sw_id
    )

    # DXF absolute board bbox (min/max) with optional layer filter
    minx_mm, miny_mm, maxx_mm, maxy_mm = board_bbox_from_dxf(args.dxf_path, args.dxf_outline_layer)

    # Process each PCB
    H_img, W_img = bgr.shape[0], bgr.shape[1]
    for idx, (iNE, iSW) in enumerate(pairs):
        ne_px, sw_px = centers[iNE], centers[iSW]
        ne_px, sw_px = label_ne_sw(ne_px, sw_px)

        # diag overlay
        cv2.circle(anno, ne_px, 6, (0,0,255), -1)
        cv2.circle(anno, sw_px, 6, (0,255,255), -1)
        cv2.line(anno, ne_px, sw_px, (255,0,255), 2)

        # center-at-midpoint using DXF dims
        left_x, right_x, top_y, bot_y, sx, sy = board_rect_from_midpoint(
            ne_px, sw_px, NE_MM, SW_MM, minx_mm, miny_mm, maxx_mm, maxy_mm
        )

        # draw center for sanity
        cx, cy = int(round(0.5*(left_x+right_x))), int(round(0.5*(top_y+bot_y)))
        cv2.circle(anno, (cx, cy), 5, (0,255,0), -1)

        # clamp vis rect
        lx_v = max(0, left_x); rx_v = min(W_img-1, right_x)
        ty_v = max(0, top_y); by_v = min(H_img-1, bot_y)
        cv2.rectangle(anno, (lx_v, ty_v), (rx_v, by_v), (255,0,0), 3)

        # warp PCB
        pcb_img, _ = warp_axis_aligned_rect(bgr, left_x, right_x, top_y, bot_y, args.pcb_out_size)
        pcb_path = os.path.join(args.outputDirectory, f"pcb_{idx}.png")
        cv2.imwrite(pcb_path, pcb_img)

        # ----- Project all STEP boxes onto warped PCB -----
        vis = pcb_img.copy()
        crops_dir = os.path.join(args.outputDirectory, f"pcb_{idx}_components")
        if args.save_component_crops:
            os.makedirs(crops_dir, exist_ok=True)

        to_px_x, to_px_y = to_px_mappers(args.pcb_out_size, minx_mm, miny_mm, maxx_mm, maxy_mm, y_up=args.dxf_y_up)

        # bottom-row auto flip?
        board_center_y = 0.5 * (top_y + bot_y)
        do_flip_180 = (board_center_y > (H_img * 0.5))

        # YOLO collection for this pcb
        yolo_boxes = []
        img_h, img_w = pcb_img.shape[:2]

        for c in comps:
            sid = c["id"]
            xmin, ymin, _, xmax, ymax, _ = c["bbox_mm"]

            if do_flip_180:
                xmin, ymin = flip_mm_180_about_board(xmin, ymin, minx_mm, miny_mm, maxx_mm, maxy_mm)
                xmax, ymax = flip_mm_180_about_board(xmax, ymax, minx_mm, miny_mm, maxx_mm, maxy_mm)
                xmin, xmax = min(xmin, xmax), max(xmin, xmax)
                ymin, ymax = min(ymin, ymax), max(ymin, ymax)

            # project to warped px (sorted, clamped, min-size guard)
            x0 = int(round(to_px_x(min(xmin, xmax))))
            x1 = int(round(to_px_x(max(xmin, xmax))))
            y0 = int(round(to_px_y(min(ymin, ymax))))
            y1 = int(round(to_px_y(max(ymin, ymax))))
            x0, x1 = sorted((max(0, x0), max(0, x1)))
            y0, y1 = sorted((max(0, y0), max(0, y1)))
            x0 = min(img_w - 1, x0); x1 = min(img_w - 1, x1)
            y0 = min(img_h - 1, y0); y1 = min(img_h - 1, y1)
            if x1 == x0 and x1 < img_w - 1: x1 += 1
            if y1 == y0 and y1 < img_h - 1: y1 += 1

            # draw overlay
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 255), 2)
            cv2.putText(vis, f"id:{sid}", (x0, max(0, y0-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,200,255), 1, cv2.LINE_AA)

            # optional per-component crop
            if args.save_component_crops:
                MINW, MINH = 4, 4
                xa, ya, xb, yb = x0, y0, x1, y1
                if (xb - xa) < MINW and (xa + MINW) < img_w: xb = xa + MINW
                if (yb - ya) < MINH and (ya + MINH) < img_h: yb = ya + MINH
                if (xb > xa) and (yb > ya):
                    crop = pcb_img[ya:yb, xa:xb]
                    cv2.imwrite(os.path.join(crops_dir, f"comp_{sid}.png"), crop)

            # YOLO export: keep only switch ids
            if args.export_yolo and (sid in SWITCH_IDS):
                yolo = _bbox_xyxy_to_yolo(x0, y0, x1, y1, img_w, img_h)
                if yolo is not None:
                    cxn, cyn, wn, hn = yolo
                    # single-class by default (index 0)
                    yolo_boxes.append((0, cxn, cyn, wn, hn))

        # save overlays
        cv2.imwrite(os.path.join(args.outputDirectory, f"pcb_{idx}_with_components.png"), vis)

        # YOLO: copy image + write labels
        if args.export_yolo:
            fname = f"pcb_{idx}.png"
            cv2.imwrite(os.path.join(YOLO_IMG_DIR, fname), pcb_img)
            with open(os.path.join(YOLO_LAB_DIR, f"pcb_{idx}.txt"), "w", encoding="utf-8") as f:
                for cls_idx, cxn, cyn, wn, hn in yolo_boxes:
                    f.write(f"{cls_idx} {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f}\n")

    cv2.imwrite(os.path.join(args.outputDirectory, "panel_annotated_mm.png"), anno)


if __name__ == "__main__":
    main()
