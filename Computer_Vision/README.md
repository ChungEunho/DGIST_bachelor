# 🧠 Introduction to Computer Vision – Course Projects

This directory contains selected projects from the *Introduction to Computer Vision* course, focusing on key topics such as image segmentation, filtering, and camera calibration.

---

## 📌 Project 1 – Image Preprocessing & Edge Detection

### 🔹 Letter Boundary Extraction
- Download `logo2_pr1.png` from LMS.
- Extract only the **letter boundaries** from the image.

### 🔹 Image Restoration – Cars
- Download `cars_pr1.png` from LMS.
- Apply **noise reduction** and **contrast enhancement** techniques.
- Restore the image to closely resemble the original.

---

## 📌 Project 2 – Orange Detection: Largest & Smallest

Given an image containing multiple oranges:
- Segment each orange using basic image processing techniques:
  - Thresholding  
  - Morphological operations  
  - Region labeling  
- Identify the **largest** and **smallest** orange based on area.
- Assume the oranges are approximately circular.
- Apply **perspective correction** if necessary to account for distortion.

---

## 📌 Project 3 – Camera Calibration & Custom Marker Measurement

- Calibrate a camera using either a smartphone or webcam.
- Design and print your own **custom marker** (not using existing solutions like ArUco).
- Attach one marker to each corner of a rectangular object.
- Capture 4 images, each focusing on a different corner.

### ✨ Your code should:
- Detect the markers.
- Output the **coordinates of the 4 corners**.
- Compute the **lengths of all 4 sides**.
- Analyze and discuss the **conditions that lead to the most accurate measurements**.

### 📝 Submission includes:
- 4 captured images  
- Source code  
- A 2-page report (due: **June 3, Tuesday**)

---

