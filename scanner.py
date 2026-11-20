
import cv2
import numpy as np

def detect_document(image):
    original_height, original_width = image.shape[:2]

    # Resize large images for better detection speed
    scale = min(1.0, 1200 / max(original_height, original_width))

    if scale < 1:
        resized = cv2.resize(
            image, None, fx=scale, fy=scale
        )
    else:
        resized = image.copy()

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 120)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Try multiple detection methods
    methods = [edges]

    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    methods.append(adaptive)

    best_contour = None
    best_area = 0

    image_area = resized.shape[0] * resized.shape[1]

    for method in methods:
        contours, _ = cv2.findContours(
            method,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True
        )

        for contour in contours[:30]:
            area = cv2.contourArea(contour)

            if area < image_area * 0.03:
                continue

            perimeter = cv2.arcLength(contour, True)

            approx = cv2.approxPolyDP(
                contour, 0.02 * perimeter, True
            )

            if (
                len(approx) == 4
                and cv2.isContourConvex(approx)
                and area > best_area
            ):
                best_area = area
                best_contour = approx

    if best_contour is not None:
        corners = best_contour.reshape(4, 2).astype("float32")
        corners = corners / scale
        return corners.astype("int32"), edges

    return None, edges

def order_points(points):
    points = np.array(points, dtype="float32")
    ordered = np.zeros((4, 2), dtype="float32")

    total = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    ordered[0] = points[np.argmin(total)]
    ordered[2] = points[np.argmax(total)]
    ordered[1] = points[np.argmin(diff)]
    ordered[3] = points[np.argmax(diff)]

    return ordered

def crop_document(image, corners):
    rect = order_points(corners)
    tl, tr, br, bl = rect

    width = int(max(
        np.linalg.norm(br - bl),
        np.linalg.norm(tr - tl)
    ))

    height = int(max(
        np.linalg.norm(tr - br),
        np.linalg.norm(tl - bl)
    ))

    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(
        rect, destination
    )

    return cv2.warpPerspective(
        image, matrix, (width, height)
    )

def enhance_document(image):
    # Improve contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([
        enhanced_l, a_channel, b_channel
    ])

    enhanced = cv2.cvtColor(
        enhanced_lab, cv2.COLOR_LAB2BGR
    )

    # Reduce noise
    enhanced = cv2.bilateralFilter(
        enhanced, 9, 75, 75
    )

    # Sharpen text and edges
    blurred = cv2.GaussianBlur(
        enhanced, (0, 0), 2
    )

    sharpened = cv2.addWeighted(
        enhanced, 1.5, blurred, -0.5, 0
    )

    return sharpened
