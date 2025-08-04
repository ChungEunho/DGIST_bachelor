import numpy as np
from PIL import Image as pilimg

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

def limit_2d(arr2d, minv, maxv):
    arr2d = np.clip(arr2d, minv, maxv)
    return arr2d

def mmn_gimg(gimg, minv=0, maxv=0):
    if minv == 0 and maxv == 0:
        minv = np.min(gimg)
        maxv = np.max(gimg)
    buff2d = 255.0 * (gimg - minv) / (maxv - minv)
    buff2d = np.round(buff2d)
    return buff2d

def z_standard(gimg, xstd):
    mean = gimg.mean()
    std = gimg.std()
    buff2d = (gimg - mean) / std
    buff2d = limit_2d(buff2d, -xstd, xstd)
    buff2d = mmn_gimg(buff2d, -xstd, xstd)
    return buff2d

def mask_filtering(img2d, mask):
    row, col = img2d.shape
    buff2d = np.zeros((row, col))
    mr, mc = mask.shape 
    hs = mr // 2
    for i in range(hs, row - hs):
        for j in range(hs, col - hs):
            sum_val = 0.0
            for p in range(-hs, hs + 1):
                for q in range(-hs, hs + 1):
                    sum_val += img2d[i + p, j + q] * mask[p + hs, q + hs]
            buff2d[i, j] = sum_val
    return buff2d

def median_filtering(img2d, ms):
    row, col = img2d.shape
    buff2d = np.zeros((row, col))
    hs = ms // 2 
    for i in range(hs, row - hs):
        for j in range(hs, col - hs):
            temp = [img2d[i + p, j + q] for p in range(-hs, hs + 1) for q in range(-hs, hs + 1)]
            temp.sort()
            buff2d[i, j] = temp[len(temp) // 2]
    return buff2d

def imagefft(img2d, ifcentering):
    imfft = np.fft.fft2(img2d)
    if ifcentering == "centering":
        imfft = np.fft.fftshift(imfft)
    return imfft

def imageifft(img2dx, ifcentering):
    if ifcentering == "centering":
        img2dx = np.fft.ifftshift(img2dx)
    ifft = np.fft.ifft2(img2dx)
    return ifft
            

def histo_eq(gimg):
    # 입력 이미지의 픽셀 값을 0~255로 클립합니다.
    gimg = np.clip(gimg, 0, 255)
    
    row, col = gimg.shape
    buff2d = np.zeros((row, col), dtype=np.uint8)
    histo = np.zeros(256, dtype=int)
    pdf = np.zeros(256, dtype=float)
    cdf = np.zeros(256, dtype=float)

    for i in range(row):
        for j in range(col):
            histo[int(gimg[i, j])] += 1

    for i in range(256):
        pdf[i] = histo[i] / (row * col)

    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf[i]

    for i in range(row):
        for j in range(col):
            buff2d[i, j] = round(255.0 * cdf[int(gimg[i, j])])

    return buff2d


def z_standard(gimg, xstd):
    mean = gimg.mean()
    std = gimg.std()
    buff2d = (gimg - mean) / std
    buff2d = limit_2d(buff2d, -xstd, xstd)
    buff2d = mmn_gimg(buff2d, -xstd, xstd)
    return buff2d

def fgaussian(ix, std):
    row, col = np.shape(ix)
    gmask = np.zeros((row, col))  # 일단 전체 0으로 초기화

    hr = row // 2
    hc = col // 2

    for i in range(-hr, hr):
        for j in range(-hc, hc):
            if (np.abs(j) == 5 and np.abs(i) == 5):
                gmask[hr + i][hc + j] = 0
            else:
                gmask[hr + i, hc + j] = np.exp(-(i**2 + j**2) / (2.0 * std**2))
    ix = ix * gmask
    return ix
# --------------------------------------------------
# 추가 함수: 그레이스케일 변환 함수
# --------------------------------------------------
def convert_to_grayscale(img):
    if len(img.shape) == 3:
        return np.dot(img[...,:3], [0.299, 0.587, 0.114])
    else:
        return img
