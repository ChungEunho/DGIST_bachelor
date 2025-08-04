import os
import json
import numpy as np
from PIL import Image as pilimg
import math
from math import pi
from collections import deque

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

def region_growing_label(inimg, th_rg):
    row, col = inimg.shape
    reg   = np.zeros((row, col, 2), float)
    reginfo = np.zeros((row*col+1, 2), float)
    rct = 0
    th = th_rg

    for i in range(row):
        for j in range(col):
            reg[i,j,1] = inimg[i,j]
            
    def merge(i,j,ii,jj,pl):  # it does not perform merging between regions except that mergeregion does only for upper and left regions, which should be implemented later.
        nonlocal rct, reg, reginfo

        reg[i][j][0]=pl

        if reg[ii][jj][0]==pl: 
            reginfo[int(pl)][0]+=1
            reginfo[int(pl)][1]=(reginfo[int(pl)][1]*(reginfo[int(pl)][0]-1)+reg[i][j][1])/reginfo[int(pl)][0]
        
        if reg[ii][jj][0]!=pl: # in case of second merging after merging once with upper or left pixel.
            old=reg[ii][jj][0]
            reg[ii][jj][0]=pl
            reginfo[int(pl)][0]+=1
            reginfo[int(pl)][1]=(reginfo[int(pl)][1]*(reginfo[int(pl)][0]-1)+reg[ii][jj][1])/reginfo[int(pl)][0]
            reginfo[int(old)][0]-=1
            
            if reginfo[int(old)][0]<=0:
                reginfo[int(old)][0]=0
                reginfo[int(old)][1]=0 
            else:
                reginfo[int(old)][1]=(reginfo[int(old)][1]*(reginfo[int(old)][0]+1)-reg[ii][jj][1])/reginfo[int(old)][0]
            
            mergeregion(old,pl)
        
        reg[i][j][1]=reginfo[int(reg[i][j][0])][1]
        
    def mergeregion(l1, l2):
        nonlocal reg, reginfo, rct, row, col
        l1=int(l1)
        l2=int(l2)
        
        if l1 == l2 or reginfo[l1][1]-reginfo[l2][1] > th:
            #print(f"Merging {l1} and {l2} failed.\n-Diff is",reginfo[l1][1]-reginfo[l2][1])
            return 0
        else:
            if l1 < l2:
                m,n=l1,l2
            else:
                m,n=l2,l1

            reginfo[m][0] += reginfo[n][0]
            reginfo[m][1] = (reginfo[m][1]*reginfo[m][0]+reginfo[n][1]*reginfo[n][0])/(reginfo[m][0]+reginfo[n][0])

            reginfo[n][0]=0
            reginfo[n][1]=0
            #print(f"Merging {l1} and {l2} happened")

            for u in range(row):
                for v in range(col):
                    if  reg[u][v][0]==n:
                            reg[u][v][0] = m
                            reg[u][v][1] = reginfo[m][1]
        
            return 1

    def separate(i,j):
        nonlocal inimg, rct, reg, reginfo
        rct+=1
        reg[i][j][0]=rct
        reginfo[int(rct)][0]+=1
        reginfo[int(rct)][1]=inimg[i][j]
    
    def relabeling():
        nonlocal reg,row, col, rct, reginfo
        ct=0
        reglabel = np.full((row*col,3),0.0)

        for i in range(rct+1):
            if reginfo[i][0]!=0:
                ct=ct+1
                reglabel[ct][0]=reginfo[i][0]
                reglabel[ct][1]=reginfo[i][1]
                reglabel[ct][2]=i # to save old label

        for i in range(rct+1):
                reginfo[i][0]=reglabel[i][0]
                reginfo[i][1]=reglabel[i][1]

        newreg=np.full((row,col,2),0.0)

        for i in range(row):
            for j in range(col):
                ol=reg[i][j][0]
                for k in range(rct+1):
                    if reglabel[k][2]==ol:
                        newreg[i][j][0]=k
                        newreg[i][j][1]=reglabel[k][1]

        for i in range(row):
            for j in range(col):
                reg[i][j][0]=newreg[i][j][0]
                reg[i][j][1]=newreg[i][j][1]

    # 8-방향 대신 위→왼 순서로 region growing
    for i in range(row):
        for j in range(col):
            rowmerge = colmerge = 0
            # 위쪽
            if i>0 and abs(reg[i,j,1] - reginfo[int(reg[i-1,j,0]),1]) <= th:
                merge(i,j, i-1,j, reg[i-1,j,0])
                rowmerge = 1
            # 왼쪽
            if j>0 and abs(reg[i,j,1] - reginfo[int(reg[i,j-1,0]),1]) <= th:
                if not rowmerge:
                    merge(i,j, i,j-1, reg[i,j-1,0])
                    colmerge = 1
                else:
                    # 이미 위쪽과 합쳐졌는데, 왼쪽 라벨이 다르면
                    if (reg[i-1,j,0] != reg[i,j-1,0] and
                        abs(reg[i,j,1] - reginfo[int(reg[i,j-1,0]),1]) <= th):
                        merge(i,j, i,j-1, reg[i-1,j,0])
                        colmerge = 1
                # 왼쪽 후 위쪽 가능해짐
                if i>0 and not rowmerge and abs(reginfo[int(reg[i,j,0]),1] - reginfo[int(reg[i-1,j,0]),1]) <= th:
                    merge(i,j, i-1,j, reg[i,j-1,0])
                    rowmerge = 1

            # 둘 다 못 합쳤으면 새 region
            if not (rowmerge or colmerge):
                separate(i,j)

    # 라벨 재정렬
    relabeling()

    label_map = reg[:,:,0].astype(int)
    return label_map, reginfo

def disk(radius):
    Y, X = np.ogrid[-radius:radius+1, -radius:radius+1]
    return (X**2 + Y**2) <= radius**2

def erode(mask, se):
    h, w = mask.shape
    rad = se.shape[0]//2
    out = np.zeros_like(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        ok = True
        for dy in range(-rad, rad+1):
            for dx in range(-rad, rad+1):
                if se[dy+rad, dx+rad]:
                    yy, xx = y+dy, x+dx
                    if not (0 <= yy < h and 0 <= xx < w and mask[yy,xx]):
                        ok = False
                        break
            if not ok:
                break
        if ok:
            out[y,x] = True
    return out

def dilate(mask, se):
    h, w = mask.shape
    rad = se.shape[0]//2
    out = np.zeros_like(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        for dy in range(-rad, rad+1):
            for dx in range(-rad, rad+1):
                if se[dy+rad, dx+rad]:
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < h and 0 <= xx < w:
                        out[yy,xx] = True
    return out

def opening(mask, radius):
    se = disk(radius)
    return dilate(erode(mask, se), se)

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

def read_image(img_name):
    im = pilimg.open(img_name)
    cimg = np.array(im)
    col, row = im.size
    return cimg, row, col

def save_image(img, name):
    img = np.uint8(img)
    im = pilimg.fromarray(img)
    im.save(name)
    return im

def morphologyf(image, index):
    row, col = image.shape
    # 배경을 0, 전경을 255로 가정
    buffer = np.full((row, col), 0, dtype=np.uint8)
    background = 0

    # 5×5 structuring element (diamond)
    se = np.array([
        (0,0,1,0,0),
        (0,1,1,1,0),
        (1,1,1,1,1),
        (0,1,1,1,0),
        (0,0,1,0,0)
    ], dtype=bool)
    se_size = 5
    se_area = se.sum()  # =13

    offset = se_size // 2
    for i in range(row - se_size + 1):
        for j in range(col - se_size + 1):
            count = 0
            # SE가 전경(255)에 걸리는 픽셀 수 세기
            for di in range(se_size):
                for dj in range(se_size):
                    if se[di, dj] and image[i+di, j+dj] == 255:
                        count += 1

            ci, cj = i + offset, j + offset
            if index == 0:  # erosion
                # SE 전부가 전경일 때만 전경(255)
                buffer[ci, cj] = 255 if count == se_area else background
            else:           # dilation
                # SE 중 하나라도 전경이면 전경(255)
                buffer[ci, cj] = 255 if count > 0 else background

    return buffer


if __name__ == "__main__":

    # 1) 원본 읽기
    cimg, rows, cols = read_image("orange1_pr2.png")

    # 2) 그레이스케일 변환 & [0,1] 정규화
    gray = cimg[..., :3].dot([0.2989, 0.5870, 0.1140]).astype(np.float32)
    gray /= gray.max()

    # 3) 이진화 (threshold T 설정)
    T = 0.59
    binary = (gray > T).astype(np.uint8) * 255
    save_image(binary, "orange1_01_binary.png")
    
    # 4) Closing (Dilation → Erosion): 작은 구멍 메우기 
    dilated  = morphologyf(binary, index=1)
    closed  = morphologyf(dilated,  index=0)
    save_image(closed, "orange1_02_closed.png")

    # 5) Opening (Ersosion → Dilation): 노이즈 제거
    eroded = morphologyf(closed, index=0)
    opened  = morphologyf(eroded, index=1)
    save_image(opened, "orange1_03_opened.png")
    
    im = pilimg.open("orange1_03_opened.png").convert("L")
    binary = np.array(im)
    
    # 5-1) erosion 으로 씨드 추출
    se_radius = 30
    se_diam   = 2 * se_radius + 1
    se = np.zeros((se_diam, se_diam), dtype=np.uint8)
    center = se_radius
    for i in range(se_diam):
        for j in range(se_diam):
            if (i - center)**2 + (j - center)**2 <= se_radius**2:
                se[i, j] = 1

    eroded = np.zeros_like(binary)
    for y in range(rows - se_diam + 1):
        for x in range(cols - se_diam + 1):
            window = binary[y:y+se_diam, x:x+se_diam]
            if np.all(window[se == 1] == 255):
                eroded[y + center, x + center] = 255
    save_image(eroded, "orange1_03_5_eroded.png")

    # 5-2) BFS로 씨드 레이블링
    h, w = eroded.shape
    seed_labels = np.zeros((h, w), dtype=int)
    label_count = 0
    for i in range(h):
        for j in range(w):
            if eroded[i, j] == 255 and seed_labels[i, j] == 0:
                label_count += 1
                q = deque([(i, j)])
                seed_labels[i, j] = label_count
                while q:
                    y, x = q.popleft()
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        yy, xx = y + dy, x + dx
                        if (0 <= yy < h and 0 <= xx < w
                            and eroded[yy, xx] == 255
                            and seed_labels[yy, xx] == 0):
                            seed_labels[yy, xx] = label_count
                            q.append((yy, xx))
    gray_vis = (seed_labels.astype(np.float32) / (label_count or 1) * 255).astype(np.uint8)
    save_image(gray_vis, "orange1_03_5_seed_labels_gray.png")
    visualize_colored(seed_labels, "orange1_03_5_seed_labels_color.png")
    
    counts = np.bincount(seed_labels.ravel())
    single_pixel_labels = np.where(counts == 1)[0]
    for lbl in single_pixel_labels:
        if lbl == 0:
            continue
        seed_labels[seed_labels == lbl] = 0

    # 5-3) Morphological Reconstruction (seed 확장)
    recon = seed_labels.copy()
    q = deque(zip(*np.nonzero(recon)))
    while q:
        y, x = q.popleft()
        lbl = recon[y, x]
        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
            yy, xx = y + dy, x + dx
            if (0 <= yy < h and 0 <= xx < w
                and binary[yy, xx] == 255
                and recon[yy, xx] == 0):
                recon[yy, xx] = lbl
                q.append((yy, xx))

    save_image((recon.astype(np.float32) / (label_count or 1) * 255).astype(np.uint8),
               "orange1_03_5_reconstructed.png")
    visualize_colored(recon, "orange1_03_5_reconstructed_color.png")

    # 6) 원형 객체 선별 + 크기 필터링
    recon_gray = pilimg.open("orange1_03_5_reconstructed.png").convert("L")
    recon_arr  = np.array(recon_gray, dtype=np.uint8)
    
    vals = np.unique(recon_arr)
    vals = vals[vals > 0]  # 0은 배경
    mapping = {v: i+1 for i, v in enumerate(sorted(vals))}
    recon = np.zeros_like(recon_arr, dtype=int)
    for intensity, lbl in mapping.items():
        recon[recon_arr == intensity] = lbl
    final_map = refine_regions (
        recon,
        circularity_thresh=0.7,
        min_size=300,
        opening_radius=12
    )
        
    bin_img = (final_map > 0).astype(np.uint8) * 255
    save_image(bin_img, "orange1_04_filtered_binary.png")
    visualize_colored(final_map, "orange1_05_filtered_color.png")
    print("완료: orange1_04_filtered_binary.png, orange1_05_filtered_color.png")
    
    # 7) 라벨 영역들 완벽한원형으로 채우기
    rows, cols = final_map.shape
    filled_map = np.zeros_like(final_map, dtype=int)
    labels = np.unique(final_map)
    labels = labels[labels != 0]
    for l in labels:
        ys, xs = np.where(final_map == l)
        if ys.size == 0:
            continue
        y_c = ys.mean()
        x_c = xs.mean()
        d2 = (ys - y_c)**2 + (xs - x_c)**2
        radius = np.sqrt(d2.max())
        Y, X = np.ogrid[:rows, :cols]
        circle_mask = (Y - y_c)**2 + (X - x_c)**2 <= radius**2
        filled_map[circle_mask] = l
    final_map = filled_map
    bin_filled = (final_map > 0).astype(np.uint8) * 255
    save_image(bin_filled, "orange1_06_filled_binary.png")
    visualize_colored(final_map, "orange1_07_filled_color.png")
    print("완료: orange1_06_filled_binary.png, orange1_07_filled_color.png")
    
    # 8) Smallest/Largest Orange 마스킹
    cimg, rows, cols = read_image("orange1_pr2.png")
    labels = np.unique(final_map)
    labels = labels[labels != 0] 
    areas = [(l, np.sum(final_map == l)) for l in labels]
    areas_sorted = sorted(areas, key=lambda x: x[1])
    small_label, small_area = areas_sorted[0]
    large_label, large_area = areas_sorted[-1]
    print(f"가장 작은 레이블 = {small_label}, 면적 = {small_area}")
    print(f"가장 큰   레이블 = {large_label}, 면적 = {large_area}")
    mask_small = (final_map == small_label)
    mask_large = (final_map == large_label)
    min_img = np.zeros_like(cimg)
    max_img = np.zeros_like(cimg)
    if cimg.ndim == 3:
        # 컬러
        min_img[mask_small] = cimg[mask_small]
        max_img[mask_large] = cimg[mask_large]
    else:
        # 그레이스케일
        min_img[mask_small] = cimg[mask_small]
        max_img[mask_large] = cimg[mask_large]
        
    save_image(min_img, "orange1_08_smallest_object.png")
    save_image(max_img, "orange1_09_largest_object.png")
    print("완료: orange1_08_smallest_object.png, 09_largest_object.png")
