import cv2
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RTSP URL to test
# Public test stream (works on AWS)
RTSP_URL = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"
# For local cameras (uncomment once NAT/VPN/relais is configured):
# RTSP_URL = "rtsp://alae:Myteam2025@192.168.0.175:554/live/ch1"
# RTSP_URL = "rtsp://alae:Myteam2025@192.168.0.202:554/live/ch1"

def test_rtsp_stream(rtsp_url):
    logger.info(f"Attempting to connect to {rtsp_url}")
    
    # Initialize video capture with FFmpeg backend
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # Set timeout properties (in milliseconds)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5-second connection timeout
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 5-second read timeout
    
    if not cap.isOpened():
        logger.error(f"Failed to open RTSP stream: {rtsp_url}")
        logger.error("Check if the URL is correct, the camera is online, and the network is accessible.")
        return
    
    logger.info("RTSP stream opened successfully")
    
    try:
        start_time = time.time()
        frame_count = 0
        while time.time() - start_time < 20:  # Test for 20 seconds
            success, frame = cap.read()
            if not success:
                logger.warning("Failed to read frame")
                time.sleep(1)
                continue
            
            # Log frame details
            height, width = frame.shape[:2]
            frame_count += 1
            logger.info(f"Frame {frame_count}: {width}x{height}")
            time.sleep(0.033)  # ~30 fps
            
        logger.info(f"Received {frame_count} frames in 20 seconds")
    
    except Exception as e:
        logger.error(f"Error during stream reading: {e}")
    
    finally:
        cap.release()
        logger.info("RTSP stream closed")

if __name__ == "__main__":
    test_rtsp_stream(RTSP_URL)