#include "all_header.h"

void get_i2c_device_two(YB_Pcb_Car* car, int address) {
    car->_addr = address;
    car->_device = wiringPiI2CSetup(car->_addr);
    if (car->_device == -1) {
        printf("Failed to initialize I2C device\n");
        exit(1);
    }
    printf("I2C device initialized at address 0x%X\n", car->_addr);
}

void write_array_two(YB_Pcb_Car* car, int reg, unsigned char* data, int length) {
    unsigned char buffer[length + 1];
    buffer[0] = reg;
    for (int i = 0; i < length; ++i) {
        buffer[i + 1] = data[i];
    }
    if (write(car->_device, buffer, length + 1) != length + 1) {
        printf("write_array I2C error\n");
    }
    else {
        //printf("write_array: Data written to reg %d\n", reg);
    }
}

void Ctrl_Servo(YB_Pcb_Car* car, int angle) {
    unsigned char data[2] = { 1, angle };
    write_array_two(car, 0x03, data, 2);
}



void* servo(void* arg) {

    wiringPiSetup();  // wiringPi 초기화
    YB_Pcb_Car car;
    get_i2c_device_two(&car, I2C_ADDR);
 // 제어할 서보 모터의 ID
    while (1) {
        Ctrl_Servo(&car, 15); // 왼쪽 (30도 기준)
        delay(200);
        Ctrl_Servo(&car, 30); // 왼쪽 (30도 기준)
        delay(200);
        Ctrl_Servo(&car, 45); // 앞쪽 (0도 기준)
        delay(200); 
        Ctrl_Servo(&car, 60); // 오른쪽 (-30도 기준)
        delay(200); 
        Ctrl_Servo(&car, 75); // 왼쪽 (30도 기준)
        delay(200);
        Ctrl_Servo(&car, 60); // 왼쪽 (30도 기준)
        delay(200);
        Ctrl_Servo(&car, 45); // 앞쪽 (0도 기준)
        delay(200); 

    }

    return NULL;
}
