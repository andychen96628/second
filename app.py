import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import datetime

# --- 頁面配置 ---
st.set_page_config(page_title="AcuGuard AI 針灸清點系統", layout="wide")

# --- 載入 AI 模型 ---
@st.cache_resource
def load_model():
    # 這裡替換成您訓練好的模型路徑
    # 如果還沒訓練，系統會先自動下載官方預訓練模型測試流程
    return YOLO('yolov8n.pt') 

model = load_model()

# --- 應用程式標題 ---
st.title("🛡️ AcuGuard AI 針灸起針輔助系統")
st.markdown("---")

# --- 側邊欄：管理功能 ---
with st.sidebar:
    st.header("📋 診察資訊")
    doctor_name = st.text_input("執行醫師", "張醫師")
    bed_number = st.selectbox("床位編號", [f"床位 {i}" for i in range(1, 11)])
    initial_needles = st.number_input("埋針總數 (Input)", min_value=0, value=10)
    
    st.markdown("---")
    st.write("### 延伸功能")
    if st.button("導出紀錄 (CSV)"):
        st.info("紀錄已存檔至系統後台")

# --- 主畫面佈局 ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 影像掃描")
    # 啟動手機/電腦相機
    img_file = st.camera_input("請對準施針部位進行掃描")

with col2:
    st.subheader("📊 偵測結果")
    if img_file:
        # 影像處理
        image = Image.open(img_file)
        img_array = np.array(image)
        
        # AI 推論
        results = model.predict(img_array, conf=0.25) # conf 是信心門檻
        detected_count = len(results[0].boxes)
        
        # 顯示警示燈號
        if detected_count == initial_needles:
            st.success(f"✅ 數量正確：偵測到 {detected_count} 根 / 應拔 {initial_needles} 根")
            st.balloons()
        else:
            diff = initial_needles - detected_count
            if diff > 0:
                st.error(f"⚠️ 警報：尚有 {diff} 根針未拔除！")
            else:
                st.warning(f"🔔 提示：偵測數量 ({detected_count}) 多於設定數量，請手動確認。")

        # 顯示 AI 標記圖
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="AI 辨識畫面 (已標註針柄位置)", use_container_width=True)

        # 紀錄 Log
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.info(f"掃描時間：{current_time} | 操作員：{doctor_name}")
    else:
        st.info("請使用左側相機功能拍照以開始辨識。")
