from q0 import read_image, convert_to_grayscale, mask_filtering, save_image, median_filtering, histo_eq
import numpy as np

while(1):
    extract_version = int(input("Select Extract Version for Problem 1[0/1]: "))
    if (extract_version != 0 and extract_version != 1):
        print("Invalid Number! Pick between 0 or 1!!")
    else:
        break

msize = 5
amask = np.full((msize, msize), 1 / (msize * msize))
gmask = np.zeros((msize, msize))
std = 2.0
hs = msize // 2
for i in range(-hs, hs + 1):
    for j in range(-hs, hs + 1):
        gmask[i + hs, j + hs] = (1.5 / hs) * (1.0 / (2 * 3.1416 * std)) * np.exp(-1.0 * (i * i + j * j) / (2 * std * std))

smaskx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])  
smasky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]]) 


def main():
    
    img, row, col = read_image("logo2_pr1.png")
    
    # 그레이 스케일 변환
    print("GrayScaling ...")
    gray = convert_to_grayscale(img)
    current_img = gray.copy()
    
    # 점 노이즈 제거
    print("Dot Denoising ...")
    num_iterations = 5
    for i in range(1, num_iterations + 1):
        mask = ((current_img == 0) | (current_img == 255))
        
        median_img = median_filtering(current_img, 5)
        
        current_img[mask] = median_img[mask] 
    pr = row // 2
    pc = col // 2
    padded_img = np.pad(current_img, ((pr, pr), (pc, pc)), mode="reflect")
    for _ in range(2):
        mask_padded = ((padded_img == 0) | (padded_img == 255))
        median_padded = median_filtering(padded_img, 5)
        padded_img[mask_padded] = median_padded[mask_padded]
    filtered_img = padded_img[pr:pr + row, pc:pc + col]

    # 스무딩
    print("Smoothing ... ")
    for _ in range(10):
        filtered_img = median_filtering(filtered_img, 5)

    # 히스토그램 이퀄라이제이션
    print("Histogram Equalizing ...")
    filtered_img = histo_eq(filtered_img)

    # 엣지 검출
    print("Edge Extraction Ongoing ...")
    extract_version_list = ["power", "nopower"]
    extract_version_str = extract_version_list[extract_version]
    
    sobel_x = mask_filtering(filtered_img, smaskx)
    sobel_y = mask_filtering(filtered_img, smasky)
    
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mag = np.abs(sobel_x) + np.abs(sobel_y)
    sobel_mag = np.maximum(np.abs(sobel_x), np.abs(sobel_y))
    sobel_norm = (sobel_mag - np.min(sobel_mag)) / (np.max(sobel_mag) - np.min(sobel_mag) + 1e-8)
    
    if (extract_version_str == "power"):
        gamma = 0.5 
        SLP_img = np.power(sobel_norm, gamma) * 255
        SLP_img = np.round(SLP_img)
        
        threshold_val = 92
        binary_edge = np.where(SLP_img >= threshold_val, 255, 0).astype(np.uint8)
        save_image(binary_edge, "logo2_pr1_edges_pow.png")
        print("Edge Image Saved(POWER)!!")
    
    else :
        sobel_x = mask_filtering(filtered_img, smaskx)
        sobel_y = mask_filtering(filtered_img, smasky)
        
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_mag = np.abs(sobel_x) + np.abs(sobel_y)
        sobel_mag = np.maximum(np.abs(sobel_x), np.abs(sobel_y))
        SLP_img = np.round(sobel_norm * 255)
        save_image(SLP_img, "logo2_pr1_edges_nopow.png")
        print("Edge Image Saved(NOPOWER)!!")


if __name__ == "__main__":
    main()