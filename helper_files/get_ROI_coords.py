"""
get_roi_coords.py

An interactive utility to define an ROI using sliders on a live video feed.

Instructions:
1. Configure the DISPLAY_WIDTH and DISPLAY_HEIGHT variables below.
2. Run the script. Two windows will appear: the video feed and a control panel.
3. The video will loop to give you time to adjust.
4. Use the sliders in the 'ROI Controls' window to position the red rectangle.
5. When you are satisfied with the ROI, press the 'q' key on your keyboard.
6. The final coordinates, scaled to the original video size, will be printed to the console.
"""

# --- Imports ---
import cv2

# --- Configurable Constants ---
# --- EDIT THESE VALUES FOR YOUR DISPLAY ---
DISPLAY_WIDTH = 1702
DISPLAY_HEIGHT = 1028



VIDEO_PATH = r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\data\videos\Sah w b3dha ghalt (2).mp4"
WINDOW_NAME = "Live ROI Selector"
CONTROLS_WINDOW_NAME = "ROI Controls"


def main():
    """
    Main function to run the live ROI selection tool.
    """
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file at '{VIDEO_PATH}'")
        return

    # Get original video dimensions for scaling calculations later
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create windows
    cv2.namedWindow(WINDOW_NAME)
    cv2.namedWindow(CONTROLS_WINDOW_NAME)

    # Empty function for trackbar callback
    def nothing(x):
        pass

    # Create trackbars for ROI coordinates based on the display dimensions
    cv2.createTrackbar("x1", CONTROLS_WINDOW_NAME, 0, DISPLAY_WIDTH, nothing)
    cv2.createTrackbar("y1", CONTROLS_WINDOW_NAME, 0, DISPLAY_HEIGHT, nothing)
    cv2.createTrackbar("x2", CONTROLS_WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_WIDTH, nothing)
    cv2.createTrackbar("y2", CONTROLS_WINDOW_NAME, DISPLAY_HEIGHT, DISPLAY_HEIGHT, nothing)

    print("✅ Adjust the sliders to define the ROI. Press 'q' to confirm and quit.")

    while True:
        success, frame = cap.read()
        if not success:
            # If video ends, loop back to the beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Resize the frame to the configured display size
        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        # Get current positions of the four trackbars
        x1 = cv2.getTrackbarPos("x1", CONTROLS_WINDOW_NAME)
        y1 = cv2.getTrackbarPos("y1", CONTROLS_WINDOW_NAME)
        x2 = cv2.getTrackbarPos("x2", CONTROLS_WINDOW_NAME)
        y2 = cv2.getTrackbarPos("y2", CONTROLS_WINDOW_NAME)

        # Draw the ROI rectangle on the display frame
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Display the resized frame
        cv2.imshow(WINDOW_NAME, display_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup and Final Output ---
    # Get final ROI coordinates from the sliders (relative to the resized window)
    x1_display = cv2.getTrackbarPos("x1", CONTROLS_WINDOW_NAME)
    y1_display = cv2.getTrackbarPos("y1", CONTROLS_WINDOW_NAME)
    x2_display = cv2.getTrackbarPos("x2", CONTROLS_WINDOW_NAME)
    y2_display = cv2.getTrackbarPos("y2", CONTROLS_WINDOW_NAME)

    cap.release()
    cv2.destroyAllWindows()

    # Scale the coordinates back to the original video dimensions
    x_scale = original_width / DISPLAY_WIDTH
    y_scale = original_height / DISPLAY_HEIGHT

    x1_final = int(x1_display * x_scale)
    y1_final = int(y1_display * y_scale)
    x2_final = int(x2_display * x_scale)
    y2_final = int(y2_display * y_scale)

    print("\n--- ROI Coordinates (Scaled to Original Video Size) ---")
    print(f"Original Video Size: {original_width}x{original_height}")
    print(f"Display Size:        {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    print(f"\nCopy this line into your next script:")
    print(f"ROI_COORDS = ({x1_final}, {y1_final}, {x2_final}, {y2_final})")


if __name__ == "__main__":
    main()