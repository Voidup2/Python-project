import cv2
from face_detection import detect_faces
from expression_recognition import recognize_expression

def run_app():
    cap = cv2.VideoCapture(0) #VideoCapture(0) as the name suggest this is used to capture video from the device's camera and 0 represents 
    #that the camera used to capture video is the primary cam or webcam in case of a laptop
    #here cap is an object created which is used to process images captured from the camera for face expression
    while cap.isOpened(): #this line checks if the cap object is still capturing images from webcam or not and in simple terms this is a check condition for the program to stop if the camera is turned off or some error has occurred
        ret, frame = cap.read() #in other words if camera is on, read the frames and ret is used to return the boolean values true if it detects the frame and false if it doesn't
        if not ret: #if the frames could not be captured then the loop breaks and stop capturing further frames 
            break

        faces = detect_faces(frame) #detect_faces is a function imported from face_detection file which detects facial expression and the frame is passed captured from cap.read

        for (x, y, w, h) in faces: #x,y are coordinates of the image and w,h is the weidth and height of the face bounding box
            face_region = frame[y:y+h, x:x+w] #this line extract the region of interst of the image i.e face, the frame passed is sliced into the face region only and stored in the face_region variable 
            emotion = recognize_expression(face_region) #we then pass face_region as a parameter in the recognize_expression function which detects the face and predicts the emotion,these are then stored in emotion variable

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2) #used to draw rectangle around the detected face, the parameter passed are current frame, coordinates of the rectangle, color of the rectangle and thickness of the rectangle
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)#used to display the text i.e the emotion detected and shown
           

        cv2.imshow('UP App', frame) #this will open a window with title UP app and displaying the current frame captured from the webcam

        if cv2.waitKey(1) & 0xFF == ord('q'): #this line allows to exit the window by pressing q from the keyboard
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": #this line makes the program executable only when the app.py is executed directly and not when some other file imports this function run_app() and tries to run it inside that file
    run_app()

