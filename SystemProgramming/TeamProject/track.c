#include "all_header.h"

static int control = straight;
pthread_mutex_t lock;
int met_Node = 0;
int path_length;
int* pMovements;
int trial = 0;
int evolution_trial = 0;
int spin_trial = 0;
int evolution = 0;

void get_i2c_device(YB_Pcb_Car* car, int address) {
    car->_addr = address;
    car->_device = wiringPiI2CSetup(car->_addr);
    if (car->_device == -1) {
        printf("Failed to initialize I2C device\n");
        exit(1);
    }
    printf("I2C device initialized at address 0x%X\n", car->_addr);
}

void write_array(YB_Pcb_Car* car, int reg, unsigned char* data, int length) {
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

void Ctrl_Car(YB_Pcb_Car* car, int l_dir, int l_speed, int r_dir, int r_speed) {
    //printf("Ctrl_Car called with l_dir: %d, l_speed: %d, r_dir: %d, r_speed: %d\n", l_dir, l_speed, r_dir, r_speed);
    unsigned char data[4] = { l_dir, l_speed, r_dir, r_speed };
    write_array(car, 0x01, data, 4);
}

void Car_Run(YB_Pcb_Car* car, int l_speed, int r_speed) {
    Ctrl_Car(car, 1, l_speed, 1, r_speed);
}

void Car_Stop(YB_Pcb_Car* car) {
    unsigned char data[4] = { 0, 0, 0, 0 };
    write_array(car, 0x01, data, 4);
}

void Car_Back(YB_Pcb_Car* car, int l_speed, int r_speed) {
    Ctrl_Car(car, 0, l_speed, 0, r_speed);
}

void Car_Left(YB_Pcb_Car* car, int l_speed, int r_speed) {
    Ctrl_Car(car, 0, l_speed, 1, r_speed);
}

void Car_Right(YB_Pcb_Car* car, int l_speed, int r_speed) {
    Ctrl_Car(car, 1, l_speed, 0, r_speed);
}

int read_sensor(int pin) {

    return  digitalRead(pin);
}

void perform_car_run_and_turn(YB_Pcb_Car* car, int* sensor_state, int control) {
    trial = 0;
    if (control == straight) {
        Car_Run(car, 60, 60);
        delay(150);
        spin_trial = 0;
    }
    else if (control == r_spin) {
        Car_Run(car, 60, 60);
        delay(150);
        Car_Right(car, 60, 60);
        delay(500);
        while ((*sensor_state & 0b0100) != 0b0100) {
            Car_Right(car, 40, 40);
            *sensor_state = *sensor_state + (read_sensor(SENSOR_LEFT2) << 2);
            delay(10);
        }
        delay(5);
        spin_trial = 0;

    }
    else if (control == l_spin) {
        Car_Run(car, 60, 60);
        delay(150);
        Car_Left(car, 60, 60);
        delay(500);
        while ((*sensor_state & 0b0010) != 0b0010) {
            Car_Left(car, 40, 40);
            *sensor_state = *sensor_state + (read_sensor(SENSOR_RIGHT2) << 1);
            delay(10);
        }
        spin_trial = 0;
        delay(5);
    }
    else if (control == turn) {
        Car_Run(car, 60, 60);
        delay(150);
        Car_Right(car, 60, 60);
        delay(1800);
        while ((*sensor_state & 0b0100) != 0b0100) {
            Car_Right(car, 40, 40);
            *sensor_state = *sensor_state + (read_sensor(SENSOR_LEFT2) << 2);
            delay(10);
        }

        delay(5);
        spin_trial++;
        if (spin_trial > 2) {
            printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~EVOLUTION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
            evolution = 1;
        }
    }
    else {
        Car_Back(car, 60, 60);
        delay(200);
        Car_Stop(car);
        delay(300);
        evolution_trial++;
    }
    pthread_mutex_lock(&lock);
    if (evolution_trial >2) {
        evolution = 1;
        evolution_trial = 0;
        free(pMovements);
        pMovements = NULL;
    }
    met_Node++;
    printf("Current Met Node %d\n", met_Node);
    if (met_Node == path_length - 1) {
        printf("free(pMovements)\n");
        free(pMovements);
        pMovements = NULL;
    }
    pthread_mutex_unlock(&lock);
}




void line_following(YB_Pcb_Car* car) {
    int left1, left2, right1, right2;


    int buffer;
    while (1) {
        if (evolution == 1) {
            break;
        }
        left1 = read_sensor(SENSOR_LEFT1);
        left2 = read_sensor(SENSOR_LEFT2);
        right1 = read_sensor(SENSOR_RIGHT1);
        right2 = read_sensor(SENSOR_RIGHT2);
        int sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
        int num_before_terminate = 0;
        switch (sensor_state) {
        case 0b0000:  // (H L L L): 전진 후 다른 
            while (sensor_state == 0b0000) {
                if (pMovements != NULL) {
                    pthread_mutex_lock(&lock);
                    control = pMovements[met_Node];
                    pthread_mutex_unlock(&lock);
                    perform_car_run_and_turn(car, &sensor_state, control);
                }
                else {
                    Car_Back(car, 65, 65);
                    delay(100);
                    Car_Stop(car);
                    delay(300);
                    evolution_trial++;
                }
                if (evolution_trial == 3) {
                    printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~EVOLUTION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
                    evolution = 1;
                    evolution_trial = 0;
                }

                left1 = read_sensor(SENSOR_LEFT1);
                left2 = read_sensor(SENSOR_LEFT2);
                right1 = read_sensor(SENSOR_RIGHT1);
                right2 = read_sensor(SENSOR_RIGHT2);
                sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
                if (sensor_state == 0b1000) {
                    break;
                }
                buffer = sensor_state;
            }
            break;
        case 0b1001:  // (H L L H) : 앞으로 직진
            Car_Run(car, 60, 60);
            left1 = read_sensor(SENSOR_LEFT1);
            left2 = read_sensor(SENSOR_LEFT2);
            right1 = read_sensor(SENSOR_RIGHT1);
            right2 = read_sensor(SENSOR_RIGHT2);
            sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
            buffer = sensor_state;
            break;
        case 0b1101:  // (H H L H): (H L L H ) 될때까지 조금씩 우회전하기

            while (sensor_state == 0b1101) {
                Car_Right(car, 60, 60);
                delay(5);
                left1 = read_sensor(SENSOR_LEFT1);
                left2 = read_sensor(SENSOR_LEFT2);
                right1 = read_sensor(SENSOR_RIGHT1);
                right2 = read_sensor(SENSOR_RIGHT2);
                sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
                if (sensor_state == 0b1001) {
                    break;
                }
                buffer = sensor_state;
            }
            break;
        case 0b1000:  // (H L L L): 전진 후 다른 것

            while (sensor_state == 0b1000) {
                if (pMovements != NULL) {
                    pthread_mutex_lock(&lock);
                    control = pMovements[met_Node];
                    pthread_mutex_unlock(&lock);
                    perform_car_run_and_turn(car, &sensor_state, control);
                }
                else {
                    Car_Back(car, 60, 60);
                    delay(100);
                    Car_Stop(car);
                    delay(300);
                    evolution_trial++;
                }
                if (evolution_trial == 3) {
                    printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~EVOLUTION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
                    evolution = 1;
                    evolution_trial = 0;
                }
                left1 = read_sensor(SENSOR_LEFT1);
                left2 = read_sensor(SENSOR_LEFT2);
                right1 = read_sensor(SENSOR_RIGHT1);
                right2 = read_sensor(SENSOR_RIGHT2);
                sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
                if (sensor_state == 0b1001) {
                    break;
                }
                buffer = sensor_state;
            }
            break;
        case 0b1110:  // (H H H L): (H H L H) 될때까지 조금씩 우회전하기
            while (sensor_state == 0b1110) {
                Car_Right(car, 60, 60);
                delay(5);
                left1 = read_sensor(SENSOR_LEFT1);
                left2 = read_sensor(SENSOR_LEFT2);
                right1 = read_sensor(SENSOR_RIGHT1);
                right2 = read_sensor(SENSOR_RIGHT2);
                sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
                if (sensor_state == 0b1001) {
                    break;
                }
                buffer = sensor_state;
            }
            break;
        case 0b1100:  // (H H L L): (H H L H) 될때까지 조금씩 우회전하기
            while (sensor_state == 0b1100) {
                if (pMovements != NULL) {
                    pthread_mutex_lock(&lock);
                    control = pMovements[met_Node];
                    pthread_mutex_unlock(&lock);
                    perform_car_run_and_turn(car, &sensor_state, control);
                }
                else {
                    Car_Back(car, 60, 60);
                    delay(100);
                    Car_Stop(car);
                    delay(300);
                    evolution_trial++;
                }
                if (evolution_trial == 3) {
                    printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~EVOLUTION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
                    evolution = 1;
                    evolution_trial = 0;
                }
                left1 = read_sensor(SENSOR_LEFT1);
                left2 = read_sensor(SENSOR_LEFT2);
                right1 = read_sensor(SENSOR_RIGHT1);
                right2 = read_sensor(SENSOR_RIGHT2);
                sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
                if (sensor_state == 0b1001) {
                    break;
                }
                buffer = sensor_state;
            }
            break;
        case 0b0111:  // (L H H H): (H L L H) 될때까지 조금씩 좌회전하기
            while (sensor_state == 0b0111) {
                Car_Left(car, 60, 60);
                delay(5);
                left1 = read_sensor(SENSOR_LEFT1);
                left2 = read_sensor(SENSOR_LEFT2);
                right1 = read_sensor(SENSOR_RIGHT1);
                right2 = read_sensor(SENSOR_RIGHT2);
                sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
                if (sensor_state == 0b1001) {
                    break;
                }
                buffer = sensor_state;

            }
            break;
        case 0b0001:  // (L H H H): (H L L H) 될때까지 조금씩 좌회전하기
            while (sensor_state == 0b0001) {
                if (pMovements != NULL) {
                    pthread_mutex_lock(&lock);
                    control = pMovements[met_Node];
                    pthread_mutex_unlock(&lock);
                    perform_car_run_and_turn(car, &sensor_state, control);
                }
                else {
                    Car_Back(car, 60, 60);
                    delay(100);
                    Car_Stop(car);
                    delay(300);
                    evolution_trial++;
                }
                if (evolution_trial == 3) {
                    printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~EVOLUTION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
                    evolution = 1;
                    evolution_trial = 0;
                }
                left1 = read_sensor(SENSOR_LEFT1);
                left2 = read_sensor(SENSOR_LEFT2);
                right1 = read_sensor(SENSOR_RIGHT1);
                right2 = read_sensor(SENSOR_RIGHT2);
                sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
                if (sensor_state == 0b1001) {
                    break;
                }
                buffer = sensor_state;
            }
            break;
        case 0b0011:
            while (sensor_state == 0b0011) {
                if (pMovements != NULL) {
                    pthread_mutex_lock(&lock);
                    control = pMovements[met_Node];
                    pthread_mutex_unlock(&lock);
                    perform_car_run_and_turn(car, &sensor_state, control);
                }
                else {
                    Car_Back(car, 60, 60);
                    delay(100);
                    Car_Stop(car);
                    delay(300);
                    evolution_trial++;
                }
                if (evolution_trial == 3) {
                    printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~EVOLUTION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
                    evolution = 1;
                    evolution_trial = 0;
                }
                left1 = read_sensor(SENSOR_LEFT1);
                left2 = read_sensor(SENSOR_LEFT2);
                right1 = read_sensor(SENSOR_RIGHT1);
                right2 = read_sensor(SENSOR_RIGHT2);
                sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
                if (sensor_state == 0b1001) {
                    break;
                }
                buffer = sensor_state;
            }
            break;
        case 0b1011:  // (H L H H): (H L L H) 될때까지 조금씩 좌회전하기
            while (sensor_state == 0b1011) {
                Car_Left(car, 60, 60);
                delay(5);
                left1 = read_sensor(SENSOR_LEFT1);
                left2 = read_sensor(SENSOR_LEFT2);
                right1 = read_sensor(SENSOR_RIGHT1);
                right2 = read_sensor(SENSOR_RIGHT2);
                sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
                if (sensor_state == 0b1001) {
                    break;
                }
                buffer = sensor_state;
            }
            break;
        case 0b1111: // (H H H H) : Stop
            if (num_before_terminate > 4) {

                Car_Stop(car);
                break;
            }

            if (num_before_terminate <= 4) {
                ++num_before_terminate;
                sensor_state = buffer;
                break;
            }
        default:
            sensor_state = buffer;
            break;
        }
        delay(5);
    }
}

void second_phase_following(YB_Pcb_Car* car) {
    int left1, left2, right1, right2;


    int buffer;
    while (1) {
        srand(time(NULL));
        int random_value = rand() % 2;
        left1 = read_sensor(SENSOR_LEFT1);
        left2 = read_sensor(SENSOR_LEFT2);
        right1 = read_sensor(SENSOR_RIGHT1);
        right2 = read_sensor(SENSOR_RIGHT2);
        int sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
        int num_before_terminate = 0;
        switch (sensor_state) {
        case 0b0000:  // (H L L L): 전진 후 다른 
            while (sensor_state == 0b0000) {
                if (random_value == 0) {
                    // right turn~
                    Car_Run(car, 60, 60);
                    delay(150);
                    Car_Right(car, 60, 60);
                    delay(500);
                    while ((sensor_state & 0b0100) != 0b0100) {
                        Car_Right(car, 40, 40);
                        sensor_state = sensor_state + (read_sensor(SENSOR_LEFT2) << 2);
                        delay(10);
                    }
                    delay(5);
                }
                else {
                    //left turn~
                    Car_Run(car, 60, 60);
                    delay(150);
                    Car_Left(car, 60, 60);
                    delay(500);
                    while ((sensor_state & 0b0010) != 0b0010) {
                        Car_Left(car, 40, 40);
                        sensor_state = sensor_state + (read_sensor(SENSOR_RIGHT2) << 1);
                        delay(10);
                    }

                    delay(5);
                }
                left1 = read_sensor(SENSOR_LEFT1);
                left2 = read_sensor(SENSOR_LEFT2);
                right1 = read_sensor(SENSOR_RIGHT1);
                right2 = read_sensor(SENSOR_RIGHT2);
                sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
                if (sensor_state == 0b1000) {
                    break;
                }
                buffer = sensor_state;
            }
            break;
        case 0b1001:  // (H L L H) : 앞으로 직진
            Car_Run(car, 100, 100);
            left1 = read_sensor(SENSOR_LEFT1);
            left2 = read_sensor(SENSOR_LEFT2);
            right1 = read_sensor(SENSOR_RIGHT1);
            right2 = read_sensor(SENSOR_RIGHT2);
            sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
            buffer = sensor_state;
            break;
        case 0b1101:  // (H H L H): (H L L H ) 될때까지 조금씩 우회전하기

            while (sensor_state == 0b1101) {
                Car_Right(car, 60, 60);
                delay(5);
                left1 = read_sensor(SENSOR_LEFT1);
                left2 = read_sensor(SENSOR_LEFT2);
                right1 = read_sensor(SENSOR_RIGHT1);
                right2 = read_sensor(SENSOR_RIGHT2);
                sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
                if (sensor_state == 0b1001) {
                    break;
                }
                buffer = sensor_state;
            }
            break;
        case 0b1000:  // (H L L L): 전진 후 다른 것

            while (sensor_state == 0b1000) {
                // right turn~
                Car_Run(car, 60, 60);
                delay(150);
                Car_Right(car, 60, 60);
                delay(500);
                while ((sensor_state & 0b0100) != 0b0100) {
                    Car_Right(car, 40, 40);
                    sensor_state = sensor_state + (read_sensor(SENSOR_LEFT2) << 2);
                    delay(10);
                }
                delay(5);
                left1 = read_sensor(SENSOR_LEFT1);
                left2 = read_sensor(SENSOR_LEFT2);
                right1 = read_sensor(SENSOR_RIGHT1);
                right2 = read_sensor(SENSOR_RIGHT2);
                sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
                if (sensor_state == 0b1001) {
                    break;
                }
                buffer = sensor_state;
            }
            break;
        case 0b1110:  // (H H H L): (H H L H) 될때까지 조금씩 우회전하기
            while (sensor_state == 0b1110) {
                Car_Right(car, 60, 60);
                delay(5);
                left1 = read_sensor(SENSOR_LEFT1);
                left2 = read_sensor(SENSOR_LEFT2);
                right1 = read_sensor(SENSOR_RIGHT1);
                right2 = read_sensor(SENSOR_RIGHT2);
                sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
                if (sensor_state == 0b1001) {
                    break;
                }
                buffer = sensor_state;
            }
            break;
        case 0b1100:  // (H H L L): (H H L H) 될때까지 조금씩 우회전하기
            while (sensor_state == 0b1100) {
                // right turn~
                Car_Run(car, 60, 60);
                delay(150);
                Car_Right(car, 60, 60);
                delay(500);
                while ((sensor_state & 0b0100) != 0b0100) {
                    Car_Right(car, 40, 40);
                    sensor_state = sensor_state + (read_sensor(SENSOR_LEFT2) << 2);
                    delay(10);
                }
                delay(5);
                left1 = read_sensor(SENSOR_LEFT1);
                left2 = read_sensor(SENSOR_LEFT2);
                right1 = read_sensor(SENSOR_RIGHT1);
                right2 = read_sensor(SENSOR_RIGHT2);
                sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
                if (sensor_state == 0b1001) {
                    break;
                }
                buffer = sensor_state;
            }
            break;
        case 0b0111:  // (L H H H): (H L L H) 될때까지 조금씩 좌회전하기
            while (sensor_state == 0b0111) {
                Car_Left(car, 60, 60);
                delay(5);
                left1 = read_sensor(SENSOR_LEFT1);
                left2 = read_sensor(SENSOR_LEFT2);
                right1 = read_sensor(SENSOR_RIGHT1);
                right2 = read_sensor(SENSOR_RIGHT2);
                sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
                if (sensor_state == 0b1001) {
                    break;
                }
                buffer = sensor_state;

            }
            break;
        case 0b0001:  // (L H H H): (H L L H) 될때까지 조금씩 좌회전하기
            while (sensor_state == 0b0001) {
                Car_Run(car, 60, 60);
                delay(150);
                Car_Left(car, 60, 60);
                delay(500);
                while ((sensor_state & 0b0010) != 0b0010) {
                    Car_Left(car, 40, 40);
                    sensor_state = sensor_state + (read_sensor(SENSOR_RIGHT2) << 1);
                    delay(10);
                }

                delay(5);

                left1 = read_sensor(SENSOR_LEFT1);
                left2 = read_sensor(SENSOR_LEFT2);
                right1 = read_sensor(SENSOR_RIGHT1);
                right2 = read_sensor(SENSOR_RIGHT2);
                sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
                if (sensor_state == 0b1001) {
                    break;
                }
                buffer = sensor_state;
            }
            break;
        case 0b0011:
            while (sensor_state == 0b0011) {
                Car_Run(car, 60, 60);
                delay(150);
                Car_Left(car, 60, 60);
                delay(500);
                while ((sensor_state & 0b0010) != 0b0010) {
                    Car_Left(car, 40, 40);
                    sensor_state = sensor_state + (read_sensor(SENSOR_RIGHT2) << 1);
                    delay(10);
                }

                delay(5);

                left1 = read_sensor(SENSOR_LEFT1);
                left2 = read_sensor(SENSOR_LEFT2);
                right1 = read_sensor(SENSOR_RIGHT1);
                right2 = read_sensor(SENSOR_RIGHT2);
                sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
                if (sensor_state == 0b1001) {
                    break;
                }
                buffer = sensor_state;
            }
            break;
        case 0b1011:  // (H L H H): (H L L H) 될때까지 조금씩 좌회전하기
            while (sensor_state == 0b1011) {
                Car_Left(car, 60, 60);
                delay(5);
                left1 = read_sensor(SENSOR_LEFT1);
                left2 = read_sensor(SENSOR_LEFT2);
                right1 = read_sensor(SENSOR_RIGHT1);
                right2 = read_sensor(SENSOR_RIGHT2);
                sensor_state = (left1 << 3) | (left2 << 2) | (right2 << 1) | right1;
                if (sensor_state == 0b1001) {
                    break;
                }
                buffer = sensor_state;
            }
            break;
        case 0b1111: // (H H H H) : Stop
            if (num_before_terminate > 4) {

                Car_Stop(car);
                break;
            }

            if (num_before_terminate <= 4) {
                ++num_before_terminate;
                sensor_state = buffer;
                break;
            }
        default:
            sensor_state = buffer;
            break;
        }
        delay(5);
    }
}



// 스레드에서 실행할 함수
void* threadFunction(void* arg) {

    if (wiringPiSetup() == -1) {
        printf("wiringPi setup failed\n");
        return NULL;
    }
    printf("WiringPi Car Setup Finished.\n");

    pinMode(SENSOR_LEFT1, INPUT);
    pinMode(SENSOR_LEFT2, INPUT);
    pinMode(SENSOR_RIGHT1, INPUT);
    pinMode(SENSOR_RIGHT2, INPUT);

    printf("Sensors pinMode finished.\n");

    YB_Pcb_Car car;
    get_i2c_device(&car, I2C_ADDR);

    printf("get_i2c_device finished.\n");

    printf("Car will stop for 1 seconds.\n");
    Car_Stop(&car);
    delay(1000);
    printf("Starting car....\n");
    while (1) {
        if (pMovements != NULL) {
            while (1) {
                line_following(&car);
                usleep(500000);
                if (evolution == 1) {
                    break;
                }
            }


        }
        if (evolution == 1) {
            break;
        }
        usleep(500000);
    }

    
    while (1) {
        second_phase_following(&car);
        usleep(500000);
    }
    usleep(500000);
    return NULL;
 }
    
