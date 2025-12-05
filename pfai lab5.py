import cv2

def main():
    img = cv2.imread('example.jpg')
    if img is None:
        print("Could not read image. Make sure 'example.jpg' exists.")
    else:
        cv2.imshow('Original Image', img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayscale Image', gray)
        cv2.imwrite('example_gray.jpg', gray)
        print("Saved grayscale image as example_gray.jpg")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Press 'q' to quit video capture")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow('Webcam Live', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
