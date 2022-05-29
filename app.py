from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
from datetime import datetime

def add_date_time(name, att_time, name_list, date_time_list ):
    if name not in name_list :
        name_list.append(name)
        date_time_list.append(att_time)
    else :
        if date_time_list[abs(name_list[::-1].index(name) - len(name_list)+ 1)][:10] != att_time[0:10] :
            name_list.append(name)
            date_time_list.append(att_time)
            
app=Flask(__name__)
camera = cv2.VideoCapture(0)

# Loading a sample picture 
ishaan_image = face_recognition.load_image_file("ishaan/ishaan.jpeg")
ishaan_face_encoding = face_recognition.face_encodings(ishaan_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [ishaan_face_encoding]
known_face_names = ["ishaan"]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
face_permanent_datetime = []
face_permanent_name = []

def gen_frames():  
    while True:
        success, frame = camera.read()  # reading camera frame
        if not success:
            break
        else:
            # Resizing frame 
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
           
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    add_date_time(name, now, face_permanent_name, face_permanent_datetime)
                face_names.append(name)
                    
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Drawing a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Drawing a label with a name
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result')
def result():
    name_data = face_permanent_name
    date_data = face_permanent_datetime
    return render_template('result.html', result_name = name_data, result_date = date_data, zip = zip)


if __name__=='__main__':
    app.run(debug=True)
