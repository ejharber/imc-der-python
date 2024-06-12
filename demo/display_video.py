
import cv2
import subprocess

def set_logitech_camera_settings(camera, brightness=100, contrast=100, saturation=100, sharpness=100, exposure=0):
    # Set brightness
    subprocess.run(f"v4l2-ctl -d /dev/video{camera} -c brightness=150", shell=True)  # Changed device path to /dev/video4

    # Set contrast
    subprocess.run(f"v4l2-ctl -d /dev/video{camera} -c contrast=120", shell=True)

    # Set saturation
    subprocess.run(f"v4l2-ctl -d /dev/video{camera} -c saturation={saturation}", shell=True)

    # Set sharpness
    subprocess.run(f"v4l2-ctl -d /dev/video{camera} -c sharpness={sharpness}", shell=True)

    # Set exposure
    subprocess.run(f"v4l2-ctl -d /dev/video{camera} -c white_balance_automatic=0", shell=True)  # Set exposure_auto_priority to 0 for manual exposure

    subprocess.run(f"v4l2-ctl -d /dev/video{camera} -c zoom_absolute=115", shell=True)  # Set exposure_auto_priority to 0 for manual exposure

    subprocess.run(f"v4l2-ctl -d /dev/video{camera} -c auto_exposure=1 -c exposure_time_absolute=140", shell=True)
    # subprocess.run(f"v4l2-ctl -d /dev/video4 -c auto_exposure=0 -c exposure_time_absolute=100", shell=True)

    subprocess.run(f"v4l2-ctl -d /dev/video{camera} -c focus_automatic_continuous=0", shell=True)

    subprocess.run(f"v4l2-ctl -d /dev/video{camera} -c exposure_dynamic_framerate=1", shell=True)



if __name__ == "__main__":
    # Open the video capture device
    camera = 0
    cap = cv2.VideoCapture(camera)  # Changed device index to 4

    # Set Logitech camera settings
    # set_logitech_camera_settings(camera)

    # Display the video feed after applying the camera settings
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        cv2.imshow('Video Feed After Settings', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and close windows
    cap.release()
    cv2.destroyAllWindows()
