import sys
import cv2
import os
import numpy as np

# ==================================================
# MODE DETECTION
# ==================================================
MODE = sys.argv[1] if len(sys.argv) > 1 else "file"

# ==================================================
# PATHS
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROI_DIR = os.path.join(BASE_DIR, "annotation")
os.makedirs(ROI_DIR, exist_ok=True)

points = []
saved = False

ZONE_VIS_DIST = 60   # purely visual (matches footfall ZONE_DIST)

# ==================================================
# MOUSE CALLBACK
# ==================================================
def mouse_click(event, x, y, flags, param):
    global saved
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) >= 2:
            print("❌ Only TWO points allowed (gate axis)")
            return
        points.append((x, y))
        saved = False
        print(f"➕ Point {len(points)} added: ({x}, {y})")

# ==================================================
# SOURCE SETUP
# ==================================================
if MODE == "rtsp":
    cam_id = sys.argv[2]
    rtsp_url = sys.argv[3]
    OUTPUT_FILE = os.path.join(ROI_DIR, f"{cam_id}_roi.txt")
    cap = cv2.VideoCapture(rtsp_url)

elif MODE == "webcam":
    cam_id = sys.argv[2]
    device_index = int(sys.argv[3])
    OUTPUT_FILE = os.path.join(ROI_DIR, f"{cam_id}_roi.txt")
    cap = cv2.VideoCapture(device_index)

else:
    video_path = input("Enter FULL VIDEO PATH: ").strip()
    if not os.path.exists(video_path):
        print("❌ Video not found")
        sys.exit(1)

    base = os.path.splitext(os.path.basename(video_path))[0]
    OUTPUT_FILE = os.path.join(ROI_DIR, f"{base}_roi.txt")
    cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Cannot open video source")
    sys.exit(1)

ret, frame = cap.read()
if not ret:
    print("❌ Failed to read first frame")
    sys.exit(1)

h, w = frame.shape[:2]

# ==================================================
# WINDOW SETUP
# ==================================================
WINDOW = "Gate ROI Annotation (2 Points)"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, w, h)
cv2.setMouseCallback(WINDOW, mouse_click)

print("\n🖱 Controls:")
print("  Left Click → Add 2 points (gate axis)")
print("  S → Save ROI")
print("  R → Reset")
print("  Q → Quit")

# ==================================================
# MAIN LOOP
# ==================================================
while True:
    draw = frame.copy()

    for p in points:
        cv2.circle(draw, p, 6, (0, 0, 255), -1)

    if len(points) == 2:
        p1, p2 = points

        # Draw gate axis
        cv2.line(draw, p1, p2, (255, 0, 0), 2)

        # Compute normal (MATCHES footfall.py)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        normal = np.array([-dy, dx], dtype=float)
        normal /= np.linalg.norm(normal)

        # Midpoint
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

        # POS arrow
        arrow_end = (
            int(mid[0] + normal[0] * 50),
            int(mid[1] + normal[1] * 50)
        )
        cv2.arrowedLine(draw, mid, arrow_end, (0, 255, 0), 2)
        cv2.putText(draw, "POS →", arrow_end,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw zone guides (visual only)
        neg_shift = (-normal[0] * ZONE_VIS_DIST, -normal[1] * ZONE_VIS_DIST)
        pos_shift = ( normal[0] * ZONE_VIS_DIST,  normal[1] * ZONE_VIS_DIST)

        neg_p1 = (int(p1[0] + neg_shift[0]), int(p1[1] + neg_shift[1]))
        neg_p2 = (int(p2[0] + neg_shift[0]), int(p2[1] + neg_shift[1]))

        pos_p1 = (int(p1[0] + pos_shift[0]), int(p1[1] + pos_shift[1]))
        pos_p2 = (int(p2[0] + pos_shift[0]), int(p2[1] + pos_shift[1]))

        cv2.line(draw, neg_p1, neg_p2, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(draw, pos_p1, pos_p2, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(draw, "NEG ZONE", neg_p1,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(draw, "POS ZONE", pos_p1,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if saved:
        cv2.putText(draw, "SAVED", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.putText(draw,
                "2 points = gate | S:Save | R:Reset | Q:Quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2)

    cv2.imshow(WINDOW, draw)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        points.clear()
        saved = False
        print("🔄 ROI reset")

    elif key == ord('s'):
        if len(points) != 2:
            print("❌ Exactly 2 points required")
            continue

        with open(OUTPUT_FILE, "w") as f:
            f.write("TYPE: LINE\n")
            for x, y in points:
                f.write(f"{x / w:.6f},{y / h:.6f}\n")

        saved = True
        print(f"💾 ROI saved → {OUTPUT_FILE}")

    elif key == ord('q'):
        break

# ==================================================
# CLEANUP
# ==================================================
cap.release()
cv2.destroyAllWindows()
print("✅ ROI annotation finished")
