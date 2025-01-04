#include "all_header.h"

int my_index;
int start_row;

int main(int argc, char *argv[]) {//MAIN THREAD
    //===================SETIING FOR SERVER CONNECTION===================
    
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <IP> <PORT>\n", argv[0]);
        return 1;
    }

    const char* server_ip = argv[1];
    int server_port = atoi(argv[2]);
    my_index = atoi(argv[3]);
    start_row = atoi(argv[4]);
    
    struct sockaddr_in serv_addr;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("Socket creation error");
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(server_port);

    if (inet_pton(AF_INET, server_ip, &serv_addr.sin_addr) <= 0) {
        perror("Invalid address/ Address not supported");
        return -1;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("Connection Failed");
        return -1;
    }

    printf("Connected to server %s:%d\n", server_ip, server_port);
    //-------------------------SERVER CONNECTION CPMPLETE--------------


    //============THREAD for QR_CODE RECEVING AND DATA SEND
    pthread_t qr_thread;
    pthread_create(&qr_thread,NULL,decodeQRCodeThread,NULL);

    //============THREAD1 FOR RECEIVING DATA FROM SERVER: update DGIST--> MAP and ENEMY=============
    pthread_t recv_thread;
    pthread_create(&recv_thread, NULL, receiveUpdates, NULL);
    
    //===========Thread3 for claculating MAX_POINT and PATH=========================================
    pthread_t alg_thread;
    pthread_create(&alg_thread,NULL,Run_Algorithm,NULL);

    //============THread2 for CAR_RUN===============================================================
    pthread_t run_thread;
    pthread_create(&run_thread,NULL,threadFunction,NULL);

    //============THread4 for SERVO===============================================================
    //pthread_t servomotor_thread;
    //pthread_create(&servomotor_thread,NULL,servo,NULL);

    // 메인 스레드는 모든 작업이 끝날 때까지 각 스레드의 종료를 기다림
    pthread_join(qr_thread, NULL);
    pthread_join(recv_thread, NULL);
    pthread_join(alg_thread, NULL);
    pthread_join(run_thread, NULL);
    //pthread_join(servomotor_thread, NULL);

    // 자원 해제
    close(sock);
    if (updatedDgist) {
        free(updatedDgist);
    }

    return 0;
}
