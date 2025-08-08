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
import cv2  # OpenCV library for video processing and GUI elements (windows, trackbars).

# --- Configurable Constants ---
# --- EDIT THESE VALUES FOR YOUR DISPLAY ---
# The desired width of the video display window for comfortable interaction.
DISPLAY_WIDTH = 1702
# The desired height of the video display window.
DISPLAY_HEIGHT = 1028


# Path to the source video file on which the ROI will be defined.
VIDEO_PATH = r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\data\videos\Sah w b3dha ghalt (2).mp4"
# Name for the main window that displays the video feed.
WINDOW_NAME = "Live ROI Selector"
# Name for the secondary window that contains the control sliders.
CONTROLS_WINDOW_NAME = "ROI Controls"


def main():
    """
    Main function to run the live ROI selection tool.
    """
    # Create a VideoCapture object to read from the specified video file.
    cap = cv2.VideoCapture(VIDEO_PATH)
    # Check if the video file was opened successfully. If not, print an error and exit.
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file at '{VIDEO_PATH}'")
        return

    # Get the original video dimensions. This is crucial for scaling the final coordinates correctly.
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create the main video window and the separate controls window.
    cv2.namedWindow(WINDOW_NAME)
    cv2.namedWindow(CONTROLS_WINDOW_NAME)

    # Define an empty callback function. `createTrackbar` requires a function to be called
    # when the slider value changes. Since we read the values directly in the loop, we don't need it to do anything.
    def nothing(x):
        pass

    # Create four trackbars (sliders) for the ROI coordinates (top-left x1,y1 and bottom-right x2,y2).
    # The sliders' range is based on the display dimensions for intuitive control.
    cv2.createTrackbar("x1", CONTROLS_WINDOW_NAME, 0, DISPLAY_WIDTH, nothing)
    cv2.createTrackbar("y1", CONTROLS_WINDOW_NAME, 0, DISPLAY_HEIGHT, nothing)
    cv2.createTrackbar("x2", CONTROLS_WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_WIDTH, nothing)
    cv2.createTrackbar("y2", CONTROLS_WINDOW_NAME, DISPLAY_HEIGHT, DISPLAY_HEIGHT, nothing)

    print("✅ Adjust the sliders to define the ROI. Press 'q' to confirm and quit.")

    # Start the main loop to process video frames and user input.
    while True:
        # Read a single frame from the video.
        success, frame = cap.read()
        # If 'success' is False, the end of the video is reached.
        if not success:
            # Loop the video by resetting the frame position to the beginning.
            # This gives the user unlimited time to adjust the ROI.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Resize the frame from its original size to the configured display size for viewing.
        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        # Get the current integer positions of the four trackbars from the controls window.
        x1 = cv2.getTrackbarPos("x1", CONTROLS_WINDOW_NAME)
        y1 = cv2.getTrackbarPos("y1", CONTROLS_WINDOW_NAME)
        x2 = cv2.getTrackbarPos("x2", CONTROLS_WINDOW_NAME)
        y2 = cv2.getTrackbarPos("y2", CONTROLS_WINDOW_NAME)

        # Draw the ROI rectangle on the display frame using the current slider values.
        # This provides immediate visual feedback. The color is BGR (Blue, Green, Red).
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red rectangle with thickness 2.

        # Show the resized frame (with the ROI) in the main window.
        cv2.imshow(WINDOW_NAME, display_frame)

        # Wait for a key press for 1 millisecond. This is necessary for imshow to update the window.
        # Check if the pressed key is 'q'.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # If 'q' is pressed, break out of the main loop.
            break

    # --- Cleanup and Final Output ---
    # This code runs after the loop has been broken.
    # Get the final positions of the sliders one last time.
    x1_display = cv2.getTrackbarPos("x1", CONTROLS_WINDOW_NAME)
    y1_display = cv2.getTrackbarPos("y1", CONTROLS_WINDOW_NAME)
    x2_display = cv2.getTrackbarPos("x2", CONTROLS_WINDOW_NAME)
    y2_display = cv2.getTrackbarPos("y2", CONTROLS_WINDOW_NAME)

    # Release the video capture object to free up resources.
    cap.release()
    # Close all OpenCV windows.
    cv2.destroyAllWindows()

    # --- Scaling Logic ---
    # Calculate the scaling factor for both width and height. This will be used to convert
    # the coordinates from the display size back to the original video's resolution.
    x_scale = original_width / DISPLAY_WIDTH
    y_scale = original_height / DISPLAY_HEIGHT

    # Apply the scaling factors to the final coordinates from the sliders.
    # Cast the results to integers, as pixel coordinates must be whole numbers.
    x1_final = int(x1_display * x_scale)
    y1_final = int(y1_display * y_scale)
    x2_final = int(x2_display * x_scale)
    y2_final = int(y2_display * y_scale)

    # Print the final, scaled coordinates to the console in a user-friendly format.
    print("\n--- ROI Coordinates (Scaled to Original Video Size) ---")
    print(f"Original Video Size: {original_width}x{original_height}")
    print(f"Display Size:        {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    print(f"\nCopy this line into your next script:")
    # The final output is formatted as a Python tuple, ready for copy-pasting.
    print(f"ROI_COORDS = ({x1_final}, {y1_final}, {x2_final}, {y2_final})")


# Standard Python entry point. Ensures `main()` is called only when the script is executed directly.
if __name__ == "__main__":
    main()