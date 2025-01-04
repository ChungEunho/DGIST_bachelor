
# 컴파일러 설정
CC = gcc
CXX = g++

# 컴파일 옵션
CFLAGS = -Wall
LDFLAGS = -lpthread -lwiringPi `pkg-config --libs opencv4`

# 타겟 파일명
TARGET = my_program

# 소스 파일 목록
CPPSRCS = qr_decoder.cpp
CSRCS = alg.c main.c track.c client.c servomotor.c


# 오브젝트 파일 목록
COBJS = $(CSRCS:.c=.o)
CPPOBJS = $(CPPSRCS:.cpp=.o)

# 기본 타겟
all: $(TARGET)

# 타겟 빌드 규칙
$(TARGET): $(COBJS) $(CPPOBJS)
	$(CXX) -o $@ $(COBJS) $(CPPOBJS) $(LDFLAGS)

# 오브젝트 파일 빌드 규칙
%.o: %.cpp
	$(CXX) -c -o $@ $< $(CFLAGS) `pkg-config --cflags opencv4`
%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

# 클린 규칙
clean:
	rm -f $(TARGET) $(COBJS) $(CPPOBJS)

# 포스 클린 규칙
.PHONY: clean
