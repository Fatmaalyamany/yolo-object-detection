from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

# -----------------------
# 🎯 Load YOLOv8 model
# -----------------------
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# -----------------------
# 🖼️ Draw bounding boxes
# -----------------------
def draw_boxes(frame, results, selected_classes):
    boxes = results.boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    class_names = [model.names[i] for i in class_ids]

    for box, cls_name, conf in zip(xyxy, class_names, confidences):
        if cls_name in selected_classes:
            x1, y1, x2, y2 = box
            label = f'{cls_name} {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6, color=(255, 0, 0), thickness=2)
    return frame

# -----------------------
# 🧠 Streamlit UI
# -----------------------
st.title('🔍 YOLOv8 Object Detection')

source_type = st.radio("Choose input source:", ["Image", "Video", "Webcam"])

upload = None
if source_type in ["Image", "Video"]:
    upload = st.file_uploader('📁 Upload your file', type=['png', 'jpeg', 'jpg', 'mp4'])

if source_type == "Image" and upload is not None:
    image = Image.open(upload).convert("RGB")
    image_np = np.array(image)

    results = model(image_np)[0]
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    class_names = [model.names[i] for i in class_ids]
    unique_classes = sorted(set(class_names))

    selected_classes = st.multiselect("✅ Select classes to display", unique_classes, default=unique_classes)

    image_with_boxes = draw_boxes(image_np.copy(), results, selected_classes)
    st.image(image_with_boxes, caption="📷 Detected Image", use_column_width=True)

elif source_type == "Video" and upload is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(upload.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    # Read one frame to detect available classes
    ret, frame = cap.read()
    if not ret:
        st.error("⚠️ Could not read video.")
        cap.release()
    else:
        results = model(frame)[0]
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        class_names = [model.names[i] for i in class_ids]
        unique_classes = sorted(set(class_names))

        selected_classes = st.multiselect("✅ Select classes to display", unique_classes, default=unique_classes)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

        st.info("⏳ Processing video frame by frame...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]
            frame = draw_boxes(frame, results, selected_classes)
            stframe.image(frame, channels="BGR", )

        cap.release()

elif source_type == "Webcam":
    run_webcam = st.checkbox("📷 Start Webcam Detection")

    if run_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        st.info("🔴 Running webcam...")

        # Read one frame to detect classes
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Could not read from webcam.")
            cap.release()
        else:
            results = model(frame)[0]
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            class_names = [model.names[i] for i in class_ids]
            unique_classes = sorted(set(class_names))

            selected_classes = st.multiselect("✅ Select classes to display", unique_classes, default=unique_classes)

            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)[0]
                frame = draw_boxes(frame, results, selected_classes)
                stframe.image(frame, channels="BGR", )

                run_webcam = st.checkbox("📷 Start Webcam Detection", value=True)

        cap.release()
        st.success("✅ Webcam stopped.")
