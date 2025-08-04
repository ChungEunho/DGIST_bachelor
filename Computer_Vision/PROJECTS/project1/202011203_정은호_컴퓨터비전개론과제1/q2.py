
from q0 import read_image, convert_to_grayscale, save_image, imagefft, imageifft, median_filtering, histo_eq, fgaussian
import numpy as np
def main():
    # 1. 이미지 로드 및 그레이스케일 변환
    img, _, _ = read_image("cars_pr.png")
    gray = convert_to_grayscale(img)
    filtered_img = gray

    # 2. 스무딩
    print("Smoothing image...")
    filtered_img = median_filtering(filtered_img, 5)

    # 3. FFT (centering)
    fft_img = imagefft(filtered_img, "centering")

    # 4. 중심 좌표 계산
    row, col = fft_img.shape
    hr = row // 2
    hc = col // 2

    for dx in [-5, 5]:
        for dy in [-5, 5]:
            fft_img[hr + dy, hc + dx] = 0

    # 6. 복원 (Inverse FFT)
    restored_img = imageifft(fft_img, "centering")
    restored_img = np.abs(restored_img)

    # 7. 히스토그램 이퀄라이제이션 및 저장
    histo_img = histo_eq(restored_img)
    save_image(histo_img, "cars_restored.png")

if __name__ == "__main__":
    main()
