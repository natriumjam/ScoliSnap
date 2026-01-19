import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import tempfile

st.set_page_config(
    page_title='ScoliSnap',
    page_icon='🩻',
    layout='centered'
)

st.title("🩻 ScoliSnap", text_alignment="center")
st.markdown("#### Let us check your back bone!", text_alignment="center")

st.markdown('### ScoliSnap Guide:')
st.markdown('1. Take a photo of your back (without clothes on)')
st.markdown('2. Make sure you photo only consist of your upper-body (shoulder to waist)')
st.markdown('3. It is recommended to have a good contrast solid background')
st.markdown('photo example:')
image_example_path = 'example_image.png'
st.image(image_example_path)

file = st.file_uploader("Upload yout photo here.", type=["jpg", "png", "jpeg"])

def detect_spine_curve(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    img = cv2.resize(img, (600, 800))
    original = img.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)

    blurred = cv2.medianBlur(enhanced_gray, 5)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)) 
    
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    processed_mask = cv2.dilate(opened, kernel, iterations=2)

    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    spine_contour = None
    max_area = 0
    img_center_x = img.shape[1] // 2
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w)
        area = cv2.contourArea(cnt)
        
        if area > 500 and aspect_ratio > 3 and abs(x - img_center_x) < 150:
            if area > max_area:
                max_area = area
                spine_contour = cnt

    if spine_contour is not None:
        cv2.drawContours(img, [spine_contour], -1, (0, 255, 0), 2)
        
        points = spine_contour.reshape(-1, 2)
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        
        rows, cols = img.shape[:2]
        lefty = int((-x0 * vy/vx) + y0)
        righty = int(((cols - x0) * vy/vx) + y0)
        cv2.line(img, (cols-1, righty), (0, lefty), (255, 0, 0), 2)

        sorted_points = points[points[:,1].argsort()]
        x_vals = sorted_points[:, 0]
        y_vals = sorted_points[:, 1]
        
        if len(y_vals) > 0:
            z = np.polyfit(y_vals, x_vals, 2)
            p = np.poly1d(z)
            
            curve_points = []
            for y in range(min(y_vals), max(y_vals), 5):
                x = int(p(y))
                curve_points.append((x, y))
            
            cv2.polylines(img, [np.array(curve_points)], False, (0, 0, 255), 3)

    titles = ['Original', 'CLAHE Enhanced', 'Threshold Mask', 'Result']
    images = [original, enhanced_gray, processed_mask, img]

    plt.figure(figsize=(10, 8))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        if i == 3 or i == 0:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    st.pyplot(plt)

def analyze_scoliosis_silhouette(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return

    height, width = img.shape[:2]
    img = cv2.resize(img, (600, int(600 * height / width)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        st.warning("No body contour detected. Please try a clearer background.")
        return None


    body_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(body_contour)

    img_h, img_w = img.shape[:2]
    upper_limit = int(img_h * 0.45)
    lower_limit = int(img_h * 0.20)

    shoulder_zone = []
    for p in hull:
        x, y = p[0]
        if lower_limit < y < upper_limit:
            shoulder_zone.append((x, y))

    mid_x = img_w // 2
    left_side = [p for p in shoulder_zone if p[0] < mid_x]
    right_side = [p for p in shoulder_zone if p[0] > mid_x]

    if len(left_side) < 1 or len(right_side) < 1:
        print("Shoulder detection failed - try clearer background or alignment.")
        return

    left_shoulder = min(left_side, key=lambda p: p[1])
    right_shoulder = min(right_side, key=lambda p: p[1])

    dy = right_shoulder[1] - left_shoulder[1]
    dx = right_shoulder[0] - left_shoulder[0]
    angle = abs(math.degrees(math.atan2(dy, dx)))
    pixel_diff = abs(left_shoulder[1] - right_shoulder[1])

    angle_threshold = 6 
    pixel_threshold = 25 

    if angle > angle_threshold or pixel_diff > pixel_threshold:
        diagnosis = "THIS **IS** SCOLIOSIS"
        color = (0, 0, 255)
    else:
        diagnosis = "THIS IS **NOT** SCOLIOSIS"
        color = (0, 255, 0)

    output = img.copy()
    cv2.drawContours(output, [hull], -1, (0, 255, 0), 2)
    cv2.circle(output, left_shoulder, 8, (255, 0, 0), -1)
    cv2.circle(output, right_shoulder, 8, (255, 0, 0), -1)
    cv2.line(output, left_shoulder, right_shoulder, (255, 0, 0), 2)

    cv2.putText(output, f"Angle: {angle:.2f} deg", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(output, f"Pixel Diff: {pixel_diff} px", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(output, diagnosis, (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
    return output

if file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(file.read())
        temp_path = tmp.name


    output = analyze_scoliosis_silhouette(temp_path)

    if output is not None:
        st.image(output, channels="BGR")
    else:
        st.error("Failed to analyze image. Please try a clearer photo with better lighting and background.")
