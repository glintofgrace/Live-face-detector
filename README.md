# Live-face-detector
# it generates a pop-up (seen on taskbar) which records for 30 second and then detects how many faces where there

#people counter using face detection
import cv2
import time

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

start_time = time.time()
frame_count = 0
faces_per_frame = []

# Run for 30 seconds
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show count on the frame
    cv2.putText(frame, f"Count: {len(faces)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("People Counter", frame)

    # Save the number of faces in this frame
    faces_per_frame.append(len(faces))
    frame_count += 1

    # Stop after 10 seconds
    if time.time() - start_time > 30:
        break

    # Wait a little so window can update
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print results
print(f"Total frames processed: {frame_count}")
print(f"Faces detected per frame: {faces_per_frame}")
print(f"Average faces per frame: {sum(faces_per_frame)/frame_count:.2f}")
[Live face.ipynb](https://github.com/user-attachments/files/22640020/Live.face.ipynb)
