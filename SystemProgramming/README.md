# 🧠 System Programming Projects @ DGIST (2025)

This repository contains assignments and the term project completed for the *System Programming* course at DGIST.  
Projects are implemented using C/C++ and evaluated on a **Raspberry Pi-based robot (Raspbot)**.

---

## 📂 Contents

- [Assignment 2: Optimized Image Convolution in C](#assignment-2-optimized-image-convolution-in-c)
- [Term Project: Raspbot Game (Line Tracing + QR Recognition + Networking)](#term-project-raspbot-game-line-tracing--qr-recognition--networking)

---

## 📌 Assignment 2: Optimized Image Convolution in C

### 🔧 Goal
Optimize the performance of a given convolution filter that processes BMP images with 3×3 kernels.

### 📝 Task Summary

- Modify the provided `filter_baseline()` implementation to create a more efficient `filter_optimized()` function in `hw2.c`.
- Focus on improving performance using system-level optimization (cache/memory access, loop unrolling, etc.)
- Evaluate performance and speedup **on Raspberry Pi using gcc -O0** (no compiler optimization).
- The output must match the original functionality exactly.

### ⚙️ Constraints

- Input image format: 24-bit BMP  
- Convolution applied to each RGB channel independently  
- No multi-threading or SIMD (e.g., NEON) allowed  
- Must support arbitrary 3×3 kernels  
- Assume image width is a multiple of 32 pixels

### 📤 Submission

- Submit optimized `hw2.c` and a short report on optimization strategy
- **Deadline**: June 14 (Fri), 11:59:59 PM (strict)

---

## 🎮 Term Project: Raspbot Game (Line Tracing + QR Code + Networking)

A semester-long team project based on a **real-time, competitive robot game** using Raspbots.  
The robot must move along lines, recognize QR codes, communicate with a server, collect items, and set traps to win.

---

### 🧱 Project Structure

#### 🔹 Part 1: Line Tracer in C

- Implement line tracing behavior using IR sensors
- Use GPIO and optionally WiringPi for Raspberry Pi control
- Build system must include a `Makefile`
- Output binary name: `linetracer`

#### 🔹 Part 2: QR Code Recognition in C++

- Use OpenCV (v4.0.0 or higher) to read QR codes from the Raspbot’s camera
- Convert decoded data (e.g., `"14"`) into coordinate values `(row=1, col=4)`
- Output binary name: `qrrecognition`
- Use `pkg-config --cflags --libs opencv4` for compilation

---

### 🌐 Part 3: Multiplayer Game with Server Communication

- **Map**: 4×4 grid (25 intersections), each with a QR code
- **Items**: Randomly spawn at intersections; give score when collected
- **Traps**: Players can set traps at intersections to sabotage opponent
- **Goal**: Earn more score than your opponent in ~2 minutes of game time

#### 🛰 Communication Protocol

- Communicate via **TCP socket** with server (localhost or remote)
- Server provides:
  - Full map state, item/trap updates
  - Opponent's location
- Client must send:
  - Player location `(x, y)`
  - Action (`0`: move only, `1`: set trap)
- Use provided `server.h` and defined `DGIST` and `ClientAction` structs

#### 📌 Initial Game Condition
