import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import datetime

# --- 頁面配置 ---
st.set_page_config(page_title="針灸 AI 清點助手", layout="centered")

# --- AI 模型載入邏輯 ---
@st.cache_resource
def load_model():
    # 如果你有訓練好的 best.pt，請放進 GitHub 並把路徑改為 'best.pt'
    custom_model_path = 'best.pt'
    
    if os.path.exists(custom_model_path):
        st.sidebar.success("✅ 載入自訂模型：best.pt")
        return YOLO(custom_model_path)
    else:
        st.sidebar.warning("⚠️ 找不到 best.pt，使用通用測試模型")
        return YOLO('yolov8n.pt') 

model = load_model()

# --- 介面設計 ---
st.title("🛡️ 針灸安全偵測系統")
st.write("透過 AI 影像辨識預防漏拔針問題")

# 側邊欄：紀錄功能
with st.sidebar:
    st.header("診間資訊")
    doctor_id = st.text_input("執行人員", "張醫師")
    target_count = st.number_input("應拔針總數", min_value=1, value=5)
    st.info("尚未上傳自訂模型前，AI 會辨識照片中的人或物體作為測試。")

# 主功能：相機拍攝
img_file = st.camera_input("請對準施針部位拍照")

if img_file:
    # 影像處理
    image = Image.open(img_file)
    img_array = np.array(image)
    
    # AI 偵測
    results = model.predict(img_array, conf=0.25)
    detected_count = len(results[0].boxes)
    
    # 結果判斷
    st.subheader(f"偵測結果：{detected_count} 根")
    
    if detected_count == target_count:
        st.success("✅ 數量相符，清點完成。")
        st.balloons()
    else:
        diff = target_count - detected_count
        if diff > 0:
            st.error(f"❌ 警報：數量不符！少偵測到 {diff} 根針。")
        else:
            st.warning(f"🔔 提示：偵測數量 ({detected_count}) 多於設定值，請人工確認。")
    
    # 顯示標記畫面
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="AI 偵測畫面 (標記框)", use_container_width=True)
    
    # 存檔存證 (模擬)
    st.caption(f"紀錄時間：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
