# 必要なモジュールをインポート
import requests
import base64
import json
from flask import Flask, render_template, Response
import cv2
import threading

# Flaskアプリケーションを初期化
app = Flask(__name__)

# カメラの映像をリアルタイムで取得する関数
def get_camera_frame():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        img_file = base64.b64encode(buffer).decode('utf-8')
        response = send_request(img_file)

        json_dict = json.loads(response.text)
        if 'faces' in json_dict and json_dict['faces']:
            emotions = json_dict['faces'][0]['attributes']['emotion']
            print(emotions)  # 表情推定結果をコンソールに出力（ここをWebページに表示する）

        _, jpg_frame = cv2.imencode('.jpg', frame)
        frame_bytes = jpg_frame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# 「Face++」に対してリクエストを送る関数（前述のものと同じ）
def send_request(img_file):
    endpoint = 'https://api-us.faceplusplus.com'
    response = requests.post(
        endpoint + '/facepp/v3/detect',
        {
            'api_key': "SK7WRvwNYjP5cVHQqzKNcUEU1J7PzxX3",  # ご自身の「API Key」を入力する
            'api_secret': "9ZLk4Teaxa0l1USu-EuCfJ_Sgv6YbdyN",  # ご自身の「API Secret」を入力する
            'image_base64': img_file,
            'return_attributes': 'emotion'
        }
    )
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(get_camera_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
