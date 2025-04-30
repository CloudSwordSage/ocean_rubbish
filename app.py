import os
import time
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from onnx_model import YOLOOnnx, draw_detections, classes
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# 初始化YOLO模型
model = YOLOOnnx('./quantized_best.onnx', conf_thres=0.3, iou_thres=0.5)

# 摄像头控制相关变量
camera_active = False
camera = None
capture_thread = None
frame = None
lock = threading.Lock()


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def generate_frames():
    global camera, frame, lock
    while camera_active:
        with lock:
            if camera and camera.isOpened():
                success, img = camera.read()
                if not success:
                    break
                try:
                    # 进行目标检测
                    boxes, scores, class_ids = model(img)
                    onnx_img = img.copy()
                    annotated_frame = draw_detections(onnx_img, boxes, scores, class_ids)
                    ret, buffer = cv2.imencode('.jpg', annotated_frame)
                    frame = buffer.tobytes()
                except Exception as e:
                    print(f"检测错误: {str(e)}")
                    break
            else:
                break
        time.sleep(0.03)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    # 修改video_feed的生成器函数
    def generate():
        global frame, lock
        while camera_active:
            with lock:  # 添加锁
                if frame:
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                else:
                    yield b''  # 返回空帧避免阻塞
            time.sleep(0.03)
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/control_camera', methods=['POST'])
def control_camera():
    global camera_active, camera, capture_thread
    action = request.json['action']

    if action == 'start':
        if not camera_active:
            try:
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    raise RuntimeError("无法打开摄像头，请检查设备连接")
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera_active = True
                capture_thread = threading.Thread(target=generate_frames)
                capture_thread.start()
                return jsonify({'status': 'camera started'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    elif action == 'stop':
        camera_active = False
        if camera is not None:
            camera.release()
            camera = None
        if capture_thread is not None:
            capture_thread.join()
            capture_thread = None
        return jsonify({'status': 'camera stopped'})
    return jsonify({'status': 'unknown action'}), 400


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    files = request.files.getlist('file')
    output = []

    for file in files:
        if file.filename == '':
            continue

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_path = os.path.join(app.config['RESULT_FOLDER'], filename)

            try:
                file.save(input_path)
                img = cv2.imread(input_path)
                if img is None:
                    raise ValueError("无效的图片文件")

                boxes, scores, class_ids = model(img)
                onnx_img = img.copy()
                annotated_frame = draw_detections(onnx_img, boxes, scores, class_ids)
                cv2.imwrite(output_path, annotated_frame)

                detections = [
                    {'class': classes[class_id], 'confidence': float(score)}
                    for _, score, class_id in zip(boxes, scores, class_ids)
                ]
                output.append({
                    'filename': filename,
                    'detections': detections,
                    'result_url': f'/results/{filename}'
                })

            except Exception as e:
                output.append({
                    'filename': filename,
                    'error': str(e)
                })
                if os.path.exists(input_path):
                    os.remove(input_path)

    return jsonify(output)


@app.route('/results/<filename>')
def get_result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
