"""
get_roi_coords.py

An interactive utility to define an ROI using sliders on a live video feed.

Instructions:
1. Run the script. Two windows will appear: the video feed and a control panel.
2. The video will loop to give you time to adjust.
3. Use the sliders in the 'ROI Controls' window to position the red rectangle.
4. When you are satisfied with the ROI, press the 'q' key on your keyboard.
5. The final coordinates will be printed to the console.
"""

# --- Imports ---
import cv2

# --- Constants ---
VIDEO_PATH = r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\videos\Sah w b3dha ghalt (2).mp4"
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

    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create windows
    cv2.namedWindow(WINDOW_NAME)
    cv2.namedWindow(CONTROLS_WINDOW_NAME)

    # Empty function for trackbar callback
    def nothing(x):
        pass

    # Create trackbars for ROI coordinates
    cv2.createTrackbar("x1", CONTROLS_WINDOW_NAME, 0, width, nothing)
    cv2.createTrackbar("y1", CONTROLS_WINDOW_NAME, 0, height, nothing)
    cv2.createTrackbar("x2", CONTROLS_WINDOW_NAME, width, width, nothing)
    cv2.createTrackbar("y2", CONTROLS_WINDOW_NAME, height, height, nothing)

    print("✅ Adjust the sliders to define the ROI. Press 'q' to confirm and quit.")

    while True:
        success, frame = cap.read()
        if not success:
            # If video ends, loop back to the beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Get current positions of the four trackbars
        x1 = cv2.getTrackbarPos("x1", CONTROLS_WINDOW_NAME)
        y1 = cv2.getTrackbarPos("y1", CONTROLS_WINDOW_NAME)
        x2 = cv2.getTrackbarPos("x2", CONTROLS_WINDOW_NAME)
        y2 = cv2.getTrackbarPos("y2", CONTROLS_WINDOW_NAME)

        # Draw the ROI rectangle on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Display the frame
        cv2.imshow(WINDOW_NAME, frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup and Final Output ---
    x1_final = cv2.getTrackbarPos("x1", CONTROLS_WINDOW_NAME)
    y1_final = cv2.getTrackbarPos("y1", CONTROLS_WINDOW_NAME)
    x2_final = cv2.getTrackbarPos("x2", CONTROLS_WINDOW_NAME)
    y2_final = cv2.getTrackbarPos("y2", CONTROLS_WINDOW_NAME)

    cap.release()
    cv2.destroyAllWindows()

    print("\n--- ROI Coordinates ---")
    print(f"Copy this line into your logic_v1.py script:")
    print(f"ROI_COORDS = ({x1_final}, {y1_final}, {x2_final}, {y2_final})")


if __name__ == "__main__":
    main()