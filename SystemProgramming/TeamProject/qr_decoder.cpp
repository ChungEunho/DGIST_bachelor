

#include "all_header.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <iostream>
#include <string>

// 스레드에서 실행할 함수

int sock;
void* decodeQRCodeThread(void* arg) {
    try {
        // OpenCV를 사용하여 카메라 캡처 초기화
        cv::VideoCapture cap(0); // 0은 기본 카메라를 의미합니다.
        if (!cap.isOpened()) {
            throw std::runtime_error("카메라를 열 수 없습니다.");
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 320); // 해상도를 높임
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
        cap.set(cv::CAP_PROP_FPS, 60); // 프레임 속도를 낮춤
        cap.set(cv::CAP_PROP_BRIGHTNESS, 50);
        cap.set(cv::CAP_PROP_CONTRAST, 70);
        cap.set(cv::CAP_PROP_EXPOSURE, 156);

        cv::Mat frame;
        std::string qrData;
        cv::QRCodeDetector qrDecoder;
        std::string before_xy = " ";

        while (true) {
            cap >> frame; // 카메라에서 프레임 캡처
            if (frame.empty()) {
                throw std::runtime_error("프레임을 캡처할 수 없습니다.");
            }

            cv::Mat grayFrame;
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

            // 히스토그램 평활화 적용
            cv::Mat equalizedFrame;
            cv::equalizeHist(grayFrame, equalizedFrame);

            try {
                // QR 코드 디코딩
                cv::Mat bbox, rectifiedImage;
                qrData = qrDecoder.detectAndDecode(equalizedFrame, bbox, rectifiedImage);

                if (!qrData.empty() && qrData != before_xy) {
                    // 디코딩된 QR 코드 데이터를 처리
                    before_xy = qrData;
                    sendClientAction(sock, &lock, qrData.c_str(), do_we_set_trap);
                }
            } catch (const cv::Exception& e) {
                std::cerr << "OpenCV Exception: " << e.what() << std::endl;
                // QR 코드 감지 실패 시 예외를 무시하고 계속 진행
            }

            // 프레임을 화면에 표시 (옵션)
            // cv::imshow("QR Code Scanner", frame);
            // if (cv::waitKey(30) >= 0) break; // 아무 키나 누르면 종료

        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
    return NULL;
}
