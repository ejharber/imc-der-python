
import cv2
import subprocess

def set_logitech_camera_settings(brightness=100, contrast=100, saturation=100, sharpness=100, exposure=0):
    # Set brightness
    subprocess.run(f"v4l2-ctl -d /dev/video4 -c brightness=160", shell=True)  # Changed device path to /dev/video4

    # Set contrast
    subprocess.run(f"v4l2-ctl -d /dev/video4 -c contrast=120", shell=True)

    # Set saturation
    subprocess.run(f"v4l2-ctl -d /dev/video4 -c saturation={saturation}", shell=True)

    # Set sharpness
    subprocess.run(f"v4l2-ctl -d /dev/video4 -c sharpness={sharpness}", shell=True)

    # Set exposure
    subprocess.run(f"v4l2-ctl -d /dev/video4 -c white_balance_automatic=0", shell=True)  # Set exposure_auto_priority to 0 for manual exposure

    subprocess.run(f"v4l2-ctl -d /dev/video4 -c zoom_absolute=115", shell=True)  # Set exposure_auto_priority to 0 for manual exposure

    subprocess.run(f"v4l2-ctl -d /dev/video4 -c auto_exposure=1 -c exposure_time_absolute=250", shell=True)
    # subprocess.run(f"v4l2-ctl -d /dev/video4 -c auto_exposure=0 -c exposure_time_absolute=100", shell=True)

    subprocess.run(f"v4l2-ctl -d /dev/video4 -c focus_automatic_continuous=0", shell=True)

    subprocess.run(f"v4l2-ctl -d /dev/video4 -c exposure_dynamic_framerate=1", shell=True)



if __name__ == "__main__":
    # Open the video capture device
    cap = cv2.VideoCapture(4)  # Changed device index to 4

    # # Display the original video feed
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     cv2.imshow('Original Video Feed', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # Set Logitech camera settings
    set_logitech_camera_settings()

    # Display the video feed after applying the camera settings
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        cv2.imshow('Video Feed After Settings', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and close windows
    cap.release()
    cv2.destroyAllWindows()
