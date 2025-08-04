import numpy as np
import math
from math import pi
from PIL import Image as pilimg
from collections import deque
import matplotlib.pyplot as plt

import numpy as np

def visualize_colored(label_map, filename):
    rows, cols = label_map.shape
    out = np.zeros((rows, cols, 3), dtype=np.uint8)

    np.random.seed(0)
    unique = np.unique(label_map)
    cmap = {l: (0,0,0) if l==0 else tuple(np.random.randint(0,256,3))
            for l in unique}

    for l, color in cmap.items():
        out[label_map == l] = color

    save_image(out, filename)

def read_image(img_name):
    im = pilimg.open(img_name).convert("RGB")
    cimg = np.array(im)
    return cimg

def save_image(img, name):
    arr = (img * 255).astype(np.uint8) if img.dtype == bool else np.uint8(img)
    pilimg.fromarray(arr).save(name)

def disk(radius):
    Y, X = np.ogrid[-radius:radius+1, -radius:radius+1]
    return (X**2 + Y**2) <= radius**2

def erode(mask, se):
    h, w = mask.shape
    rad = se.shape[0] // 2
    out = np.zeros_like(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        ok = True
        for dy in range(-rad, rad+1):
            for dx in range(-rad, rad+1):
                if se[dy+rad, dx+rad]:
                    yy, xx = y+dy, x+dx
                    if not (0 <= yy < h and 0 <= xx < w and mask[yy, xx]):
                        ok = False
                        break
            if not ok:
                break
        if ok:
            out[y, x] = True
    return out

def dilate(mask, se):
    h, w = mask.shape
    rad = se.shape[0] // 2
    out = np.zeros_like(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        for dy in range(-rad, rad+1):
            for dx in range(-rad, rad+1):
                if se[dy+rad, dx+rad]:
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < h and 0 <= xx < w:
                        out[yy, xx] = True
    return out

def label_bitmap(mask):
    h, w = mask.shape
    labels = np.zeros((h,w), int)
    current = 0
    for i in range(h):
        for j in range(w):
            if mask[i,j] and labels[i,j] == 0:
                current += 1
                # BFS
                q = deque([(i,j)])
                labels[i,j] = current
                while q:
                    y,x = q.popleft()
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        yy, xx = y+dy, x+dx
                        if (0 <= yy < h and 0 <= xx < w 
                            and mask[yy,xx] and labels[yy,xx]==0):
                            labels[yy,xx] = current
                            q.append((yy,xx))
    return labels, current

def compute_circularity(mask):
    area = mask.sum()
    # 경계 픽셀만 남기기
    er = erode(mask, disk(1))
    boundary = mask & (~er)
    perim = boundary.sum() or 1
    return 4 * pi * area / (perim * perim)

def opening(mask, radius):
    se = disk(radius)
    return dilate(erode(mask, se), se)

def refine_regions(label_map,
                         circularity_thresh=0.7,
                         min_size=300,
                         opening_radius=5):

    final_map = np.zeros_like(label_map, dtype=int)
    next_label = 1

    for rid in np.unique(label_map):
        if rid == 0:
            continue
        region = (label_map == rid)

        # 붙어 있는 원 분리
        opened = opening(region, opening_radius)

        # 분리된 컴포넌트별 레이블링
        comps, ncomp = label_bitmap(opened)
        for cid in range(1, ncomp+1):
            comp_mask = (comps == cid)
            area = comp_mask.sum()
            if area < min_size:
                continue
            circ = compute_circularity(comp_mask)
            if circ < circularity_thresh:
                continue
            # 여기를 통과한 조각만 붙여 줌
            final_map[comp_mask] = next_label
            next_label += 1

    return final_map

def morphologyf(image, index):
    row, col = image.shape
    buffer = np.zeros((row, col), dtype=np.uint8)
    se = np.array([
        (0,0,1,0,0),
        (0,1,1,1,0),
        (1,1,1,1,1),
        (0,1,1,1,0),
        (0,0,1,0,0)
    ], dtype=bool)
    size = 5
    area = se.sum()
    off = size // 2

    for i in range(row - size + 1):
        for j in range(col - size + 1):
            cnt = 0
            for di in range(size):
                for dj in range(size):
                    if se[di, dj] and image[i+di, j+dj] == 255:
                        cnt += 1
            ci, cj = i+off, j+off
            if index == 0:   # erosion
                buffer[ci, cj] = 255 if cnt == area else 0
            else:            # dilation
                buffer[ci, cj] = 255 if cnt > 0 else 0
    return buffer

if __name__ == "__main__":

    # 1) 원본 읽기 -> 그레이스케일 -> 이진화
    cimg = read_image("orange2_pr2.png")
    gray = cimg[..., :3].dot([0.2989, 0.5870, 0.1140]).astype(np.float32)
    gray /= gray.max()
    T = 0.55
    binary = (gray > T)

    # 2) 열림/닫힘 필터링
    mask = binary.astype(np.uint8) * 255
    open1 = morphologyf(mask, index=0)
    open2 = morphologyf(open1, index=1)
    close1 = morphologyf(open2, index=1)
    mask_clean = morphologyf(close1, index=0).astype(bool)

    save_image(mask_clean, "orange2_01_mask_clean.png")

    # 3) 큰 SE로 침식하여 seeds 생성
    big_se = disk(radius=40)  # 오렌지 사이 틈보다 큰 숫자
    seeds = erode(mask_clean, big_se)
    save_image(seeds, "orange2_02_seeds.png")

    # 4) BFS로 Seed 레이블링
    h, w = seeds.shape
    labels = np.zeros((h, w), int)
    cnt = 0
    for i in range(h):
        for j in range(w):
            if seeds[i, j] and labels[i, j] == 0:
                cnt += 1
                q = deque([(i, j)])
                labels[i, j] = cnt
                while q:
                    y, x = q.popleft()
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        yy, xx = y+dy, x+dx
                        if (0 <= yy < h and 0 <= xx < w 
                            and seeds[yy, xx] and labels[yy, xx] == 0):
                            labels[yy, xx] = cnt
                            q.append((yy, xx))

    # 5) Morphological Reconstruction (seed 확장)
    recon = labels.copy()
    q = deque(zip(*np.nonzero(recon)))
    while q:
        y, x = q.popleft()
        lbl = recon[y, x]
        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
            yy, xx = y+dy, x+dx
            if (0 <= yy < h and 0 <= xx < w 
                and mask_clean[yy, xx] 
                and recon[yy, xx] == 0):
                recon[yy, xx] = lbl
                q.append((yy, xx))

    # ——— label2rgb 대신 간단 오버레이 ———
    # get_color 함수 정의
    def get_color(lbl):
        return np.array([(lbl * 37) % 256,
                         (lbl * 59) % 256,
                         (lbl * 97) % 256], dtype=np.uint8)

    # 원본 이미지 복사
    overlay = cimg.copy()
    # alpha 블렌딩 계수
    alpha = 0.5

    for y, x in zip(*np.nonzero(recon)):
        lbl = recon[y, x]
        color = get_color(lbl)
        orig = overlay[y, x].astype(np.uint16)
        blended = ((1 - alpha) * orig + alpha * color).astype(np.uint8)
        overlay[y, x] = blended

    save_image(overlay, "orange2_03_first_overlay.png")

    # 레이블맵 자체 시각화
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title("First Labels")
    plt.imshow(recon, cmap="nipy_spectral")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.title("First Overlay")
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    #plt.show()

    np.save("orange2_04_recon.npy", recon)
    
    # 6) 원형 객체 선별 + 크기 필터링
    cimg = read_image("orange2_pr2.png")
    recon = np.load("orange2_04_recon.npy")
    
    final_map = refine_regions(
        recon,
        circularity_thresh=0.7,   # 원형성 70% 이상
        min_size=300,             # 최소 300px 이상
        opening_radius=5          # opening SE 반지름 5px
    )
    
    binary_final = (final_map > 0).astype(np.uint8) * 255
    save_image(binary_final, "orange2_05_filtered_binary.png")
    visualize_colored(final_map, "orange2_06_filtered_overlay.png")
    np.save("orange2_07_final_map.npy", final_map)
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title("Filtered Labels")
    plt.imshow(final_map, cmap="nipy_spectral")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.title("Filtered Overlay")
    colored_final = read_image("orange2_06_filtered_overlay.png")
    plt.imshow(colored_final)
    plt.axis("off")
    plt.tight_layout()
    #plt.show()

    # 7’) Oblique 보정 실제 면적 계산 → radius_from_area → 완전 원으로 채우기
    final_map = np.load("orange2_07_final_map.npy")
    rows, cols = final_map.shape
    H_minus1 = rows - 1
    s_bottom, s_top = 1.0, 2.0

    areas = {}
    for l in np.unique(final_map):
        if l == 0: 
            continue
        ys, xs = np.nonzero(final_map == l)
        pixel_sizes = s_bottom + (H_minus1 - ys) / H_minus1 * (s_top - s_bottom)
        areas[l] = np.sum(pixel_sizes**2)

    radii = {l: math.sqrt(A / math.pi) for l, A in areas.items()}

    filled_map = np.zeros_like(final_map, dtype=int)
    Y, X = np.ogrid[:rows, :cols]
    for l, r in radii.items():
        ys, xs = np.nonzero(final_map == l)
        y_c, x_c = ys.mean(), xs.mean()
        avg_size = (s_bottom + s_top) / 2
        r_px = r / avg_size
        circle = (Y - y_c)**2 + (X - x_c)**2 <= r_px**2
        filled_map[circle] = l

    final_map = filled_map

    bin_filled = (final_map > 0).astype(np.uint8) * 255
    save_image(bin_filled, "orange2_08_filled_binary.png")
    visualize_colored(final_map, "orange2_09_filled_color.png")
    

    # 8) 가장 큰 두 영역을 제거하고, 나머지 중에서 가장 작은/가장 큰 영역을 선택
    areas_sorted = sorted(areas.items(), key=lambda x: x[1])
    if len(areas_sorted) > 2:
        remaining = areas_sorted[:-2]
    else:
        remaining = areas_sorted[:]  
    small_label, small_area = remaining[0]
    large_label, large_area = remaining[-1]

    print(f"Excluding top 2 largest areas, remaining smallest: L{small_label} ({small_area:.2f} mm²), "
          f"largest: L{large_label} ({large_area:.2f} mm²)")

    # 9) Smallest/Largest Orange 마스킹 & 저장

    cimg = read_image("orange2_pr2.png")

    mask_small = (final_map == small_label)
    mask_large = (final_map == large_label)

    min_img = np.zeros_like(cimg)
    max_img = np.zeros_like(cimg)

    if cimg.ndim == 3:
        min_img[mask_small] = cimg[mask_small]
        max_img[mask_large] = cimg[mask_large]
    else:
        min_img[mask_small] = cimg[mask_small]
        max_img[mask_large] = cimg[mask_large]

    save_image(min_img, "orange2_10_smallest_object.png")
    save_image(max_img, "orange2_11_largest_object.png")
    print("Saved smallest and largest masked objects: orange2_10_smallest_object.png, orange2_11_largest_object.png")