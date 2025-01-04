#ifndef ALL_HEADER_H
#define ALL_HEADER_H
//======================TOTAL INCLUDE================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <unistd.h> 
//========================ALGO INCLUDE================================
#include <stdbool.h>
#include <netinet/in.h>
//=========================CAR RUN INCLUDES============================
#include <wiringPi.h>
#include <wiringPiI2C.h>
//========================CLIENT DEFINE===============================
//========================SERVER DEFINE===============================
#define MAX_CLIENTS 2
#define _MAP_ROW 4
#define _MAP_COL 4
#define MAP_ROW (_MAP_ROW + 1)
#define MAP_COL (_MAP_COL + 1)
#define MAP_SIZE (MAP_COL * MAP_ROW);

//========================ALGO DEFINES=================================
#define MAX_QUEUE_SIZE (MAP_ROW * MAP_COL)
#define LEFT 1
#define UP 2
#define RIGHT 3
#define DOWN 4
#define l_spin 1
#define straight 2
#define r_spin 3
#define turn 4
//=====================CAR RUN DEFFINES==================================
#define I2C_ADDR 0x16
#define NUM_READINGS 10
#define SENSOR_RIGHT2 0  
#define SENSOR_RIGHT1 7  
#define SENSOR_LEFT1 2   
#define SENSOR_LEFT2 3   
//=====================STRUCTS OF SERVER================================
// 클라이언트 정보 구조체 정의
typedef struct {
    int socket;
    struct sockaddr_in address;
    int row;
    int col;
    int score;
    int bomb;
} client_info;

// 아이템 상태 열거형 정의
enum Status {
    nothing, // 0
    item, // 1
    trap // 2
};

// 아이템 구조체 정의
typedef struct {
    enum Status status;
    int score;
} Item;

// 맵의 각 교차점을 나타내는 구조체 정의
typedef struct {
    int row;
    int col;
    Item item;
} Node;

// 전체 게임 상태를 나타내는 구조체 정의
typedef struct {
    client_info players[MAX_CLIENTS];
    Node map[MAP_ROW][MAP_COL];
} DGIST;

// 클라이언트 행동 열거형 정의
enum Action {
    move, // 0
    setBomb // 1
};

// 클라이언트 행동 구조체 정의
typedef struct {
    int row;
    int col;
    enum Action action;
} ClientAction;

//=========================THREADING========================================
//------------------variables---------------------
//========================QR QODE DECODER :: update XY========================

#ifdef __cplusplus
extern "C" {
#endif
    extern int sock;
    extern pthread_mutex_t lock; 
    void sendClientAction(int sock, pthread_mutex_t* lock, const char* coordinates, int action);

    void* decodeQRCodeThread(void* arg);//THREAD FUNCTION: READING QR

#ifdef __cplusplus
}
#endif

//===========================CLIENT :: update DGIST========================================
//--------------------------VARIABLES--------------------------
extern DGIST* updatedDgist;
//-------------------------FUNCTIONS--------------------------
void* receiveUpdates(void* arg); // THREAD FUNCTION: use (FUN)updateGlobalVariables
void updateGlobalVariables(DGIST* dgist,int my_sock);

//==========================ALGORITHM :: update path=========================================
//-------------------------VARIABLES--------------------------
extern bool do_we_set_trap;
extern int my_index;
extern int start_row;
extern int met_Node;
extern int path_length;
extern int* pMovements;
extern int calculate;
//-------------------------STURCTS---------------------------
typedef struct {
    int x;
    int y;
} Point;

typedef struct {
    Point points[_MAP_ROW * _MAP_COL];
    int length;
    int score;
} Path;

typedef struct {
    Point point;
    int distance;
} QueueNode;

//-----------------------FUNCTIONS-------------------------
bool isValid(Point p, Point* points, int count);
Point* Bangaljook(int opp_x, int opp_y, int my_x, int my_y, int* count);
Point* Find_MaxScorePoint(Point* StartPoint, Point* points, int count);
void copy_path(Path* dest, Path* src);
void find_paths(int row_moves, int column_moves, Path* path, int current_score, Path* best_path, int start_x, int start_y);
Point* find_best_road(Point* StartPoint, Point* EndPoint, int* path_length);
int* getDirection(Point* road, int path_length);
int* getDirection_for_Mov(int* Dir, int path_length, int recent_head_dir);
int* getMovement(int* dirs_for_movs, int path_length);
int SetBomb_Checker(Point* currpoint, Point* opponentpoint);
void* Run_Algorithm(void* arg);//THREAD FUNCTION : FINDING PATH

//===============================CAR RUN :: RUN A CAR============================================
//----------------------STRUCTS----------------------
typedef struct {
    int _device;
    int _addr;
} YB_Pcb_Car;
//--------------------FUNCTIONS-----------------------
void get_i2c_device(YB_Pcb_Car* car, int address);
void write_array(YB_Pcb_Car* car, int reg, unsigned char* data, int length);
void Ctrl_Car(YB_Pcb_Car* car, int l_dir, int l_speed, int r_dir, int r_speed);
void Car_Run(YB_Pcb_Car* car, int l_speed, int r_speed);
void Car_Stop(YB_Pcb_Car* car);
void Car_Back(YB_Pcb_Car* car, int l_speed, int r_speed);
void Car_Left(YB_Pcb_Car* car, int l_speed, int r_speed);
void Car_Right(YB_Pcb_Car* car, int l_speed, int r_speed);
int read_sensor(int pin);
void perform_car_run_and_turn(YB_Pcb_Car* car, int* sensor_state, int control);
void line_following(YB_Pcb_Car* car);
void* threadFunction(void* arg);
//===============================SERVO :: RUN A SERVO MOTOR============================================
void get_i2c_device_two(YB_Pcb_Car* car, int address);
void write_array_two(YB_Pcb_Car* car, int reg, unsigned char* data, int length);
void Ctrl_Servo(YB_Pcb_Car* car, int angle);
void* servo(void*arg);
#endif // ALL_HEADER_H
