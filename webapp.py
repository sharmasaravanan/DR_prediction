from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from prediction import DR
import cv2
app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def predict():
    dr = DR()
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == "POST":
        if "button" in request.form:
            if request.form['button'] == "Capture":
                cam = cv2.VideoCapture(0)
                while True:
                    ret, frame = cam.read()
                    org = frame.copy()
                    cv2.putText(frame, "Press 'Esc' to close or 'c' to capture.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
                    cv2.imshow("Image Capturing System", frame)
                    if not ret:
                        break
                    k = cv2.waitKey(1)
                    if k % 256 == 27:
                        break
                    elif k % 256 == 99:
                        cv2.imwrite("testImage.jpg", org)
                        break
                cam.release()
                cv2.destroyAllWindows()
                name, score = dr.prediction("testImage.jpg")
                kwargs = {'name': name, 'score': score}
                return render_template('index2.html', **kwargs)
        else:
            image = request.files.get('imgup')
            image.save('./' + secure_filename(image.filename))
            img = image.filename
            name, score = dr.prediction(img)
            kwargs = {'name': name, 'score': score}
            return render_template('index2.html', **kwargs)


if __name__ == '__main__':
    app.run(host='0.0.0.0')