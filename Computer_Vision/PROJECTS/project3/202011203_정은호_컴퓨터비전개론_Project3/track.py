import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
import sys


def order_points_for_marker_bboxes(marker_bboxes_with_indices):
    if len(marker_bboxes_with_indices) != 4:
        return [item[0] for item in marker_bboxes_with_indices]
    sorted_by_y = sorted(marker_bboxes_with_indices, key=lambda item: item[1][1])
    top_row_candidates = sorted_by_y[:2]
    bottom_row_candidates = sorted_by_y[2:]
    top_row_sorted = sorted(top_row_candidates, key=lambda item: item[1][0])
    bottom_row_sorted = sorted(bottom_row_candidates, key=lambda item: item[1][0])
    return [
        top_row_sorted[0][0], top_row_sorted[1][0],
        bottom_row_sorted[1][0], bottom_row_sorted[0][0]
    ]

def find_internal_shape_contours_from_markers(image_path):
    image_original = cv2.imread(image_path)
    
    hsv_image = cv2.cvtColor(image_original, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])
    black_mask = cv2.inRange(hsv_image, lower_black, upper_black)
    kernel = np.ones((5, 5), np.uint8)
    black_mask_processed = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    black_mask_processed = cv2.morphologyEx(black_mask_processed, cv2.MORPH_OPEN, kernel, iterations=2)

    marker_contours, _ = cv2.findContours(black_mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marker_bboxes_with_indices_unsorted = [] 
    marker_rois_unsorted = []
    min_marker_area = 1000

    for i, cnt_marker in enumerate(marker_contours):
        perimeter_marker = cv2.arcLength(cnt_marker, True)
        approx_marker = cv2.approxPolyDP(cnt_marker, 0.02 * perimeter_marker, True)
        area_marker = cv2.contourArea(approx_marker)
        if len(approx_marker) == 4 and area_marker > min_marker_area:
            x, y, w, h = cv2.boundingRect(approx_marker)
            roi = image_original[y:y+h, x:x+w]
            marker_bboxes_with_indices_unsorted.append((i, (x, y, w, h)))
            marker_rois_unsorted.append({'roi_img': roi, 'offset_x': x, 'offset_y': y, 
                                         'original_index': i, 'marker_contour_global': approx_marker})

    
    final_ordered_marker_rois_data = []
    if len(marker_bboxes_with_indices_unsorted) == 4:
        sorted_original_indices = order_points_for_marker_bboxes(marker_bboxes_with_indices_unsorted)
        temp_roi_dict = {data['original_index']: data for data in marker_rois_unsorted}
        final_ordered_marker_rois_data = [temp_roi_dict[idx] for idx in sorted_original_indices if idx in temp_roi_dict]
    else:
        final_ordered_marker_rois_data = sorted(marker_rois_unsorted, 
                                                key=lambda x: cv2.contourArea(x['marker_contour_global']), 
                                                reverse=True)

    all_pentagon_contours_global = []
    all_rectangle_contours_global = []
    
    for marker_data in final_ordered_marker_rois_data:
        roi_img = marker_data['roi_img']
        offset_x = marker_data['offset_x']
        offset_y = marker_data['offset_y']
        if roi_img is None or roi_img.size == 0: 
            all_pentagon_contours_global.append(None) 
            all_rectangle_contours_global.append(None)
            continue

        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        internal_contours, _ = cv2.findContours(thresh_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_pentagon_contour_roi = None; pentagon_max_area = 0
        best_rectangle_contour_roi = None; rectangle_max_area = 0
        min_area_ratio = 0.01; max_area_ratio = 0.50
        roi_total_area = roi_img.shape[0] * roi_img.shape[1]
        min_abs_area = roi_total_area * min_area_ratio
        max_abs_area = roi_total_area * max_area_ratio
        
        found_pentagon_for_this_marker = False
        found_rectangle_for_this_marker = False

        for cnt_internal in internal_contours:
            area_internal = cv2.contourArea(cnt_internal)
            if not (min_abs_area < area_internal < max_abs_area): continue
            perimeter_internal = cv2.arcLength(cnt_internal, True)
            approx_internal = cv2.approxPolyDP(cnt_internal, 0.025 * perimeter_internal, True)
            num_vertices = len(approx_internal)

            if num_vertices == 5 and area_internal > pentagon_max_area:
                pentagon_max_area = area_internal
                best_pentagon_contour_roi = approx_internal
            elif num_vertices == 4:
                x_b,y_b,w_b,h_b = cv2.boundingRect(approx_internal)
                aspect_ratio = float(w_b)/h_b if h_b > 0 else 0
                if 0.05 < aspect_ratio < 20.0 and area_internal > rectangle_max_area:
                    rectangle_max_area = area_internal
                    best_rectangle_contour_roi = approx_internal
        
        if best_pentagon_contour_roi is not None:
            temp_contour = best_pentagon_contour_roi.reshape(-1, 2).copy() 
            temp_contour[:, 0] += offset_x; temp_contour[:, 1] += offset_y
            all_pentagon_contours_global.append(temp_contour.reshape(-1, 1, 2))
            found_pentagon_for_this_marker = True
        if not found_pentagon_for_this_marker: 
             all_pentagon_contours_global.append(None)

        if best_rectangle_contour_roi is not None:
            temp_contour = best_rectangle_contour_roi.reshape(-1, 2).copy()
            temp_contour[:, 0] += offset_x; temp_contour[:, 1] += offset_y
            all_rectangle_contours_global.append(temp_contour.reshape(-1, 1, 2))
            found_rectangle_for_this_marker = True
        if not found_rectangle_for_this_marker: 
             all_rectangle_contours_global.append(None)
            
    return all_pentagon_contours_global, all_rectangle_contours_global

def determine_marker_orientations(pentagon_contours_global, rectangle_contours_global):
    orientations = []
    num_markers_to_process = min(len(pentagon_contours_global), len(rectangle_contours_global))

    if len(pentagon_contours_global) != len(rectangle_contours_global):
        pass

    for i in range(num_markers_to_process):
        p_cnt = pentagon_contours_global[i]
        r_cnt = rectangle_contours_global[i]

        if p_cnt is None and r_cnt is None: orientations.append("Error: Both Missing"); continue
        elif p_cnt is None: orientations.append("Error: Pentagon Missing"); continue
        elif r_cnt is None: orientations.append("Error: Rectangle Missing"); continue
        
        try:
            M_p = cv2.moments(p_cnt.astype(np.int32)); cx_p = int(M_p["m10"]/M_p["m00"]) if M_p["m00"]!=0 else -1
            M_r = cv2.moments(r_cnt.astype(np.int32)); cx_r = int(M_r["m10"]/M_r["m00"]) if M_r["m00"]!=0 else -1
        except: orientations.append("Error: Moment Calc"); continue

        if cx_p == -1 or cx_r == -1: orientations.append("Error: CX Invalid"); continue
        if cx_p < cx_r: orientations.append('B')
        elif cx_r < cx_p: orientations.append('T')
        else: orientations.append("Error: Same CX")
    return orientations

def get_overall_image_orientation(individual_orientations):

    if not individual_orientations:
        return "Undetermined (No valid orientations)"

    valid_orientations = [ori for ori in individual_orientations if ori in ['B', 'T']]
    
    if not valid_orientations:
        return "Undetermined (No B or T found)"
        
    counts = Counter(valid_orientations)
    count_b = counts.get('B', 0)
    count_t = counts.get('T', 0)


    if count_b > count_t:
        return 'B'
    elif count_t > count_b:
        return 'T'
    else: 
        if count_b == 0 : 
            return "Undetermined (No B or T found)"
        else:
            return "Undetermined (Equal B and T)"     

def read_calibration_params(param_dir="parameters"):
    intr_path = os.path.join(param_dir, "camera_matrix.txt")
    dist_path = os.path.join(param_dir, "dist_coeffs.txt")
    if not os.path.isfile(intr_path) or not os.path.isfile(dist_path):
        intr_path = "camera_matrix.txt"
        dist_path = "dist_coeffs.txt"
    mtx = np.loadtxt(intr_path)
    dist = np.loadtxt(dist_path)
    return mtx, dist

def order_points(pts):
    """
    4개의 점을 받아 좌상단, 우상단, 우하단, 좌하단 순서로 정렬
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # 좌상단 (sum 최소)
    rect[2] = pts[np.argmax(s)] # 우하단 (sum 최대)
    diff_val = np.array([p[0] - p[1] for p in pts])
    rect[1] = pts[np.argmax(diff_val)] # 우상단
    rect[3] = pts[np.argmin(diff_val)] # 좌하단
    return rect

def detect_markers(cimg):

    output_image_visualization = cimg.copy() 
    hsv_image = cv2.cvtColor(cimg, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 70])
    black_mask = cv2.inRange(hsv_image, lower_black, upper_black)
    
    kernel = np.ones((5, 5), np.uint8)
    black_mask_processed = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    black_mask_processed = cv2.morphologyEx(black_mask_processed, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(black_mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marker_candidates = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx_poly = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
        area = cv2.contourArea(cnt)
        if len(approx_poly) == 4 and area > 1000: 
            marker_candidates.append({"contour": cnt, "approx": approx_poly, "area": area})

    sorted_markers_by_area = sorted(marker_candidates, key=lambda m: m["area"], reverse=True)
    num_to_select = 4 
    selected_markers = sorted_markers_by_area[:num_to_select]

    all_marker_corners_list = [] 
    all_marker_centers_list = [] 

    for marker_info in selected_markers: 
        points_raw = marker_info["approx"].reshape(4, 2)
        ordered_points_for_marker = order_points(points_raw.astype(np.float32))
        all_marker_corners_list.append(ordered_points_for_marker)

        center_x = np.mean(ordered_points_for_marker[:, 0])
        center_y = np.mean(ordered_points_for_marker[:, 1])
        all_marker_centers_list.append(np.array([center_x, center_y], dtype=np.float32))
        cv2.drawContours(output_image_visualization, [marker_info["contour"]], -1, (0, 255, 0), 2)
        for point in ordered_points_for_marker:
            cv2.circle(output_image_visualization, tuple(point.astype(int)), 7, (0, 0, 255), -1)
            
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output_image_visualization, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Markers & Corners ({len(all_marker_corners_list)} found)")
    plt.axis('off')
    plt.suptitle("Marker Detection Result", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    if all_marker_corners_list:
        marker_corners_unsorted = np.array(all_marker_corners_list, dtype=np.float32)
        marker_centers_unsorted = np.array(all_marker_centers_list, dtype=np.float32)
        return marker_corners_unsorted, marker_centers_unsorted, len(all_marker_corners_list)
    else:
        return None, None, 0

def estimate_marker_poses_and_sort(
        marker_corners_list_unsorted,
        marker_centers_list_unsorted,
        camera_matrix, dist_coeffs, marker_real_size_mm,
        orientation):
    
    num_markers_found = len(marker_corners_list_unsorted)
    if num_markers_found != 4:
        print(f"Error in PnP: Expected 4 markers for pose estimation, but got {num_markers_found}.")
        return None

    # 3D 객체 포인트 정의 (마커 로컬 좌표계, 중심이 원점) : 순서는 order_points 함수가 반환하는 TL, TR, BR, BL과 일치해야 함
    half = marker_real_size_mm / 2.
    if (orientation == 'B'):
        object_points_3d = np.array([
            [-half,  half, 0.0],  # Top-Left
            [ half,  half, 0.0],  # Top-Right
            [ half, -half, 0.0],  # Bottom-Right
            [-half, -half, 0.0]   # Bottom-Left
        ], dtype=np.float32)
    elif (orientation == 'T'):
        object_points_3d = np.array([
            [ half, -half, 0.0],  # Top-Left
            [-half, -half, 0.0],  # Top-Right
            [-half,  half, 0.0],  # Bottom-Right
            [ half,  half, 0.0]   # Bottom-Left
        ], dtype=np.float32)

    marker_pose_data_unsorted = []

    for i in range(num_markers_found):
        corners_2d_for_pnp = marker_corners_list_unsorted[i]
        center_2d = marker_centers_list_unsorted[i]

        # SolvePnP
        success, rvec, tvec = cv2.solvePnP(object_points_3d, corners_2d_for_pnp,
                                           camera_matrix, dist_coeffs)
        if success:
            marker_pose_data_unsorted.append({
                'center_2d': center_2d, 
                'tvec': tvec.reshape(3), 
                'rvec': rvec.reshape(3), 
                'corners_2d': corners_2d_for_pnp 
            })
        else:
            print(f"Warning: SolvePnP failed for marker with 2D center approx {center_2d}")
            return None 
            
    if len(marker_pose_data_unsorted) != 4:
        print("Error: Pose estimation did not succeed for all 4 markers.")
        return None
        
    marker_pose_data_unsorted.sort(key=lambda item: item['center_2d'][1])
    top_row_markers = sorted(marker_pose_data_unsorted[:2], key=lambda item: item['center_2d'][0])
    bottom_row_markers = sorted(marker_pose_data_unsorted[2:], key=lambda item: item['center_2d'][0])

    sorted_marker_data = [
        top_row_markers[0],    # Top-Left
        top_row_markers[1],    # Top-Right
        bottom_row_markers[1], # Bottom-Right 
        bottom_row_markers[0]  # Bottom-Left 
    ]

    sorted_tvecs = np.array([data['tvec'] for data in sorted_marker_data]) 
    print("\nSpatially Sorted 3D Marker Positions (tvecs):")
    labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    for i, data in enumerate(sorted_marker_data):
        print(f"  Marker {labels[i]} (2D center: {data['center_2d'].round(1)}): "
              f"3D Pos: {data['tvec'].round(2)} mm")
    return sorted_tvecs

def calculate_distances_between_markers(sorted_marker_tvecs_tl_tr_br_bl):

    m_tl, m_tr, m_br, m_bl = (
        sorted_marker_tvecs_tl_tr_br_bl[0], # Top-Left
        sorted_marker_tvecs_tl_tr_br_bl[1], # Top-Right
        sorted_marker_tvecs_tl_tr_br_bl[2], # Bottom-Right
        sorted_marker_tvecs_tl_tr_br_bl[3]  # Bottom-Left
    )

    dist_top_edge = np.linalg.norm(m_tl - m_tr)       # TL-TR
    dist_right_edge = np.linalg.norm(m_tr - m_br)     # TR-BR
    dist_bottom_edge = np.linalg.norm(m_br - m_bl)    # BR-BL 
    dist_left_edge = np.linalg.norm(m_bl - m_tl)      # BL-TL 
    
    edge_lengths = [dist_top_edge, dist_right_edge, dist_bottom_edge, dist_left_edge]
    
    print("\nCalculated lengths of the sides formed by markers:")
    labels = ["Top Edge (TL-TR)", "Right Edge (TR-BR)", "Bottom Edge (BR-BL)", "Left Edge (BL-TL)"]

    expected_lengths_approx = [100, 200, 100, 200] 

    for i, length in enumerate(edge_lengths):
        print(f"  Side {labels[i]}: {length:.2f} mm (Expected ~{expected_lengths_approx[i]} mm)")
    
    return edge_lengths

def main(img_original, orientation):
    mtx, dist = read_calibration_params()
    print("Camera Intrinsic Matrix (mtx):\n", mtx)
    print("Distortion Coefficients (dist):\n", dist)
    
    marker_corners_unsorted, marker_centers_unsorted, num_detected = detect_markers(img_original)

    if marker_corners_unsorted is not None and num_detected == 4:
        print(f"\nSuccessfully detected {num_detected} markers from the image.")
        MARKER_REAL_SIZE_MM = 38.0  # 마커의 실제 한 변 길이 (mm)
        
        sorted_tvecs_spatially = estimate_marker_poses_and_sort(
            marker_corners_unsorted, 
            marker_centers_unsorted, 
            mtx, dist, MARKER_REAL_SIZE_MM,
            orientation
        )
        
        if sorted_tvecs_spatially is not None:
            return calculate_distances_between_markers(sorted_tvecs_spatially)
        else:
            print("Could not obtain sorted 3D positions for all markers. Cannot calculate distances.")
            return None
    elif num_detected > 0:
        print(f"Error: Expected 4 markers for PnP, but detected {num_detected}. Cannot proceed further.")
        return None
    else:
        print("No markers detected or an error occurred in detection.")
        return None

if __name__ == '__main__':
    image_list = ["Usermarker_BL.jpeg","Usermarker_BR.jpeg","Usermarker_TL.jpeg","Usermarker_TR.jpeg"]
    usermarker_dict = {
        "Image Index 0": "usermarker_BL", "Image Index 1": "usermarker_BR",
        "Image Index 2": "usermarker_TL", "Image Index 3": "usermarker_TR"
    } 
    
    all_side_lengths = []

    for i in range(len(image_list)):
        image_filename = image_list[i]
        img_original = cv2.imread(image_filename)
        usermarker_name = usermarker_dict[f"Image Index {i}"]
        print(f"------------------------------------{usermarker_name} RESULT---------------------------------\n")
        
        print(f"Processing Image: {image_filename}")
        pentagon_contours_raw, rectangle_contours_raw = \
            find_internal_shape_contours_from_markers(image_filename) 
        valid_pentagon_contours = [cnt for cnt in pentagon_contours_raw if cnt is not None]
        valid_rectangle_contours = [cnt for cnt in rectangle_contours_raw if cnt is not None]
        if not valid_pentagon_contours and not valid_rectangle_contours: 
            print("Cannot find valid pentagon or rectangular contour in the image.")
            orientation = "Undetermined"
        else:
            individual_marker_orientations = determine_marker_orientations(
                pentagon_contours_raw, rectangle_contours_raw
            )
            
            labels = [" (Top-Left)  ", " (Top-Right) ", " (Bottom-Right)  ", " (Bottom-Left) "]
            num_orientations = len(individual_marker_orientations)

            if num_orientations > 0:
                use_spatial_labels = (num_orientations == 4 and 
                                      len(pentagon_contours_raw) == 4 and 
                                      len(rectangle_contours_raw) == 4)
            else:
                print("  Cannot determine individual marker orientation.")

            orientation = get_overall_image_orientation(individual_marker_orientations)
            print(f"  Image Orientation Determination Result: {orientation}\n")       

        edge_lengths = main(img_original, orientation)
        if edge_lengths is not None:
            all_side_lengths.append(edge_lengths)

    # 평균 계산
    if len(all_side_lengths) == 4:
        all_sides_array = np.array(all_side_lengths)  # shape (4, 4)
        avg_sides = np.mean(all_sides_array, axis=0)

        print("\n==================== FINAL AVERAGED SIDE LENGTHS ====================")
        final_labels = ["Top Edge (TL-TR)", "Right Edge (TR-BR)", "Bottom Edge (BR-BL)", "Left Edge (BL-TL)"]
        expected_lengths = [100, 200, 100, 200]
        for i in range(4):
            print(f"  {final_labels[i]}: {avg_sides[i]:.2f} mm (Expected ~{expected_lengths[i]} mm)")
        print("=====================================================================\n")
    else:
        print("\nFinal Averaged Side Lengths could not be computed. Not all images yielded valid results.\n")
