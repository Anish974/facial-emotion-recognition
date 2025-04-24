import cv2
from deepface import DeepFace
import sys

def main():
    try:
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise Exception("Failed to load face cascade classifier")

        # Start capturing video
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open video device")

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                continue

            # Convert frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(
                gray_frame, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                try:
                    # Extract the face ROI in BGR format (no conversion needed)
                    face_roi = frame[y:y + h, x:x + w]

                    # Perform emotion analysis on the face ROI
                    result = DeepFace.analyze(
                        face_roi, 
                        actions=['emotion'], 
                        enforce_detection=False,
                        detector_backend='opencv'
                    )

                    # Determine the dominant emotion
                    emotion = result[0]['dominant_emotion']

                    # Draw rectangle around face and label with predicted emotion
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, emotion, (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                except Exception as e:
                    print(f"Error processing face: {str(e)}")
                    continue

            # Display the resulting frame
            cv2.imshow('Real-time Emotion Detection', frame)

            

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
    finally:
        # Release resources
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

