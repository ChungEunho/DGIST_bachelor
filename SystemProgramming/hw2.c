#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "bmplib.h"
#include "hw2.h"

#define block_size 100
#define SCALE_FACTOR 256 // 고정 소수점 스케일링 팩터

void filter_optimized(void* args[]) {
    unsigned int width = *(unsigned int*)args[0];
    unsigned int height = *(unsigned int*)args[1];
    Pixel* input = args[2];
    Pixel* output = args[3];
    float* filter = args[4];

    int width_unrolled = width / 8 * 8; // Ensure width is multiple of 8 for loop unrolling

    for (int y = 0; y < height; ++y) {
        int rowIndex = y * width;

        // Loop unrolled by 8 for processing pixels
        for (int x = 0; x < width_unrolled; x += 8) {
            //Pixel p1, p2, p3, p4, p5, p6, p7, p8;
            double val_r1 = 0, val_r2 = 0, val_r3 = 0, val_r4 = 0;
            double val_r5 = 0, val_r6 = 0, val_r7 = 0, val_r8 = 0;
            double val_g1 = 0, val_g2 = 0, val_g3 = 0, val_g4 = 0;
            double val_g5 = 0, val_g6 = 0, val_g7 = 0, val_g8 = 0;
            double val_b1 = 0, val_b2 = 0, val_b3 = 0, val_b4 = 0;
            double val_b5 = 0, val_b6 = 0, val_b7 = 0, val_b8 = 0;

            // Loop over the filter
            for (int dy = -1; dy < 2; ++dy) {
                int ny = y + dy;
                if (ny >= 0 && ny < height) {
                    int rowIndexOffset = ny * width;
                    for (int dx = -1; dx <= 1; ++dx) {
                        int filterIndex = (dy + 1) * 3 + (dx + 1);
                        float filterValue = filter[filterIndex];
                        int nx1 = x + dx;
                        int nx2 = x + dx + 1;
                        int nx3 = x + dx + 2;
                        int nx4 = x + dx + 3;
                        int nx5 = x + dx + 4;
                        int nx6 = x + dx + 5;
                        int nx7 = x + dx + 6;
                        int nx8 = x + dx + 7;

                        if (nx1 >= 0 && nx1 < width) {
                            int inputIndex1 = rowIndexOffset + nx1;
                            val_r1 += input[inputIndex1].r * filterValue;
                            val_g1 += input[inputIndex1].g * filterValue;
                            val_b1 += input[inputIndex1].b * filterValue;
                        }
                        if (nx2 >= 0 && nx2 < width) {
                            int inputIndex2 = rowIndexOffset + nx2;
                            val_r2 += input[inputIndex2].r * filterValue;
                            val_g2 += input[inputIndex2].g * filterValue;
                            val_b2 += input[inputIndex2].b * filterValue;
                        }
                        if (nx3 >= 0 && nx3 < width) {
                            int inputIndex3 = rowIndexOffset + nx3;
                            val_r3 += input[inputIndex3].r * filterValue;
                            val_g3 += input[inputIndex3].g * filterValue;
                            val_b3 += input[inputIndex3].b * filterValue;
                        }
                        if (nx4 >= 0 && nx4 < width) {
                            int inputIndex4 = rowIndexOffset + nx4;
                            val_r4 += input[inputIndex4].r * filterValue;
                            val_g4 += input[inputIndex4].g * filterValue;
                            val_b4 += input[inputIndex4].b * filterValue;
                        }
                        if (nx5 >= 0 && nx5 < width) {
                            int inputIndex5 = rowIndexOffset + nx5;
                            val_r5 += input[inputIndex5].r * filterValue;
                            val_g5 += input[inputIndex5].g * filterValue;
                            val_b5 += input[inputIndex5].b * filterValue;
                        }
                        if (nx6 >= 0 && nx6 < width) {
                            int inputIndex6 = rowIndexOffset + nx6;
                            val_r6 += input[inputIndex6].r * filterValue;
                            val_g6 += input[inputIndex6].g * filterValue;
                            val_b6 += input[inputIndex6].b * filterValue;
                        }
                        if (nx7 >= 0 && nx7 < width) {
                            int inputIndex7 = rowIndexOffset + nx7;
                            val_r7 += input[inputIndex7].r * filterValue;
                            val_g7 += input[inputIndex7].g * filterValue;
                            val_b7 += input[inputIndex7].b * filterValue;
                        }
                        if (nx8 >= 0 && nx8 < width) {
                            int inputIndex8 = rowIndexOffset + nx8;
                            val_r8 += input[inputIndex8].r * filterValue;
                            val_g8 += input[inputIndex8].g * filterValue;
                            val_b8 += input[inputIndex8].b * filterValue;
                        }
                    }
                }
            }

            // Store the clamped pixel values directly into output array
            uint8_t* out_ptr = (uint8_t*)&output[rowIndex + x];
            out_ptr[0] = (uint8_t)(val_r1 < 0 ? 0 : (val_r1 > 255 ? 255 : val_r1));
            out_ptr[1] = (uint8_t)(val_g1 < 0 ? 0 : (val_g1 > 255 ? 255 : val_g1));
            out_ptr[2] = (uint8_t)(val_b1 < 0 ? 0 : (val_b1 > 255 ? 255 : val_b1));

            out_ptr[3] = (uint8_t)(val_r2 < 0 ? 0 : (val_r2 > 255 ? 255 : val_r2));
            out_ptr[4] = (uint8_t)(val_g2 < 0 ? 0 : (val_g2 > 255 ? 255 : val_g2));
            out_ptr[5] = (uint8_t)(val_b2 < 0 ? 0 : (val_b2 > 255 ? 255 : val_b2));

            out_ptr[6] = (uint8_t)(val_r3 < 0 ? 0 : (val_r3 > 255 ? 255 : val_r3));
            out_ptr[7] = (uint8_t)(val_g3 < 0 ? 0 : (val_g3 > 255 ? 255 : val_g3));
            out_ptr[8] = (uint8_t)(val_b3 < 0 ? 0 : (val_b3 > 255 ? 255 : val_b3));

            out_ptr[9] = (uint8_t)(val_r4 < 0 ? 0 : (val_r4 > 255 ? 255 : val_r4));
            out_ptr[10] = (uint8_t)(val_g4 < 0 ? 0 : (val_g4 > 255 ? 255 : val_g4));
            out_ptr[11] = (uint8_t)(val_b4 < 0 ? 0 : (val_b4 > 255 ? 255 : val_b4));

            out_ptr[12] = (uint8_t)(val_r5 < 0 ? 0 : (val_r5 > 255 ? 255 : val_r5));
            out_ptr[13] = (uint8_t)(val_g5 < 0 ? 0 : (val_g5 > 255 ? 255 : val_g5));
            out_ptr[14] = (uint8_t)(val_b5 < 0 ? 0 : (val_b5 > 255 ? 255 : val_b5));

            out_ptr[15] = (uint8_t)(val_r6 < 0 ? 0 : (val_r6 > 255 ? 255 : val_r6));
            out_ptr[16] = (uint8_t)(val_g6 < 0 ? 0 : (val_g6 > 255 ? 255 : val_g6));
            out_ptr[17] = (uint8_t)(val_b6 < 0 ? 0 : (val_b6 > 255 ? 255 : val_b6));

            out_ptr[18] = (uint8_t)(val_r7 < 0 ? 0 : (val_r7 > 255 ? 255 : val_r7));
            out_ptr[19] = (uint8_t)(val_g7 < 0 ? 0 : (val_g7 > 255 ? 255 : val_g7));
            out_ptr[20] = (uint8_t)(val_b7 < 0 ? 0 : (val_b7 > 255 ? 255 : val_b7));

            out_ptr[21] = (uint8_t)(val_r8 < 0 ? 0 : (val_r8 > 255 ? 255 : val_r8));
            out_ptr[22] = (uint8_t)(val_g8 < 0 ? 0 : (val_g8 > 255 ? 255 : val_g8));
            out_ptr[23] = (uint8_t)(val_b8 < 0 ? 0 : (val_b8 > 255 ? 255 : val_b8));
        }

        // 처리되지 않은 나머지 픽셀 처리
        for (int x = width_unrolled; x < width; ++x) {
            Pixel p;
            double val_r = 0;
            double val_g = 0;
            double val_b = 0;

            for (int dy = -1; dy < 2; ++dy) {
                int ny = y + dy;
                if (ny >= 0 && ny < height) {
                    int rowIndexOffset = ny * width;
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = x + dx;
                        int filterIndex = (dy + 1) * 3 + (dx + 1);
                        float filterValue = filter[filterIndex];
                        if (nx >= 0 && nx < width) {
                            int inputIndex = rowIndexOffset + nx;
                            val_r += input[inputIndex].r * filterValue;
                            val_g += input[inputIndex].g * filterValue;
                            val_b += input[inputIndex].b * filterValue;
                        }
                    }
                }
            }

            // Clamping values to be within 0 to 255
            p.r = (unsigned char)(val_r < 0 ? 0 : (val_r > 255 ? 255 : val_r));
            p.g = (unsigned char)(val_g < 0 ? 0 : (val_g > 255 ? 255 : val_g));
            p.b = (unsigned char)(val_b < 0 ? 0 : (val_b > 255 ? 255 : val_b));

            output[rowIndex + x] = p;
        }
    }
}