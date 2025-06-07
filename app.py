import streamlit as st
import requests
import json
from datetime import datetime

# Cấu hình trang
st.set_page_config(
    page_title="Chat với Gemini & Gemma",
    page_icon="🤖",
    layout="wide"
)

# CSS để làm đẹp giao diện
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        color: #1f1f1f !important;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #0d47a1 !important;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        color: #4a148c !important;
    }
    .chat-message strong {
        color: inherit !important;
        font-weight: bold;
    }
    .chat-message p, .chat-message div {
        color: inherit !important;
    }
    .timestamp {
        font-size: 0.8rem;
        color: #666 !important;
        margin-top: 0.5rem;
        opacity: 0.7;
    }
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #c62828 !important;
    }
    /* Đảm bảo text trong chat luôn có contrast tốt */
    .user-message * {
        color: #0d47a1 !important;
    }
    .assistant-message * {
        color: #4a148c !important;
    }
    .user-message .timestamp {
        color: #1565c0 !important;
    }
    .assistant-message .timestamp {
        color: #7b1fa2 !important;
    }
</style>
""", unsafe_allow_html=True)

# Tiêu đề
st.title("🤖 Chat với Google AI Models")
st.markdown("Trò chuyện với Gemini và Gemma thông qua Google AI Studio API")

# Sidebar để cấu hình
with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    # Nhập API Key
    api_key = st.text_input(
        "Google AI Studio API Key:",
        type="password",
        help="Lấy API key từ https://aistudio.google.com/"
    )
    
    # Chọn model
    model_choice = st.selectbox(
        "Chọn Model:",
        [
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash"
            # "gemma-3-27b-it"
        ],
        help="Chọn model AI bạn muốn sử dụng"
    )
    
    # Cài đặt generation config
    st.subheader("Cài đặt Model")
    temperature = st.slider("Temperature (độ sáng tạo):", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider("Max tokens:", 100, 8192, 1024, 100)
    top_p = st.slider("Top P:", 0.0, 1.0, 0.95, 0.05)
    top_k = st.slider("Top K:", 1, 100, 40, 1)
    
    # Nút xóa lịch sử
    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Thông tin
    st.markdown("---")
    st.markdown("**📋 Dependencies cần thiết:**")
    st.code("pip install streamlit requests", language="bash")

# Khởi tạo session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hàm gọi API trực tiếp
def call_gemini_api(message, model_name, api_key, temp=0.7, max_tok=1024, top_p=0.95, top_k=40):
    """
    Gọi Google AI Studio API trực tiếp sử dụng requests
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": message
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": temp,
            "topK": top_k,
            "topP": top_p,
            "maxOutputTokens": max_tok,
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                if 'content' in result['candidates'][0]:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    return "❌ Phản hồi bị chặn bởi safety filters."
            else:
                return "❌ Không nhận được phản hồi từ model."
        else:
            error_msg = f"HTTP {response.status_code}"
            try:
                error_detail = response.json()
                if 'error' in error_detail:
                    error_msg += f": {error_detail['error'].get('message', 'Unknown error')}"
            except:
                pass
            return f"❌ Lỗi API: {error_msg}"
            
    except requests.exceptions.Timeout:
        return "❌ Timeout: API không phản hồi trong 30 giây."
    except requests.exceptions.RequestException as e:
        return f"❌ Lỗi kết nối: {str(e)}"
    except Exception as e:
        return f"❌ Lỗi không xác định: {str(e)}"

# Hàm kiểm tra API key
def test_api_key(api_key, model_name):
    """
    Test API key với một câu hỏi đơn giản
    """
    test_response = call_gemini_api("Hello", model_name, api_key, temp=0.1, max_tok=10)
    return not test_response.startswith("❌")

# Kiểm tra API key
if not api_key:
    st.warning("⚠️ Vui lòng nhập Google AI Studio API Key ở sidebar để bắt đầu!")
    st.info("""
    **Hướng dẫn lấy API Key:**
    1. Truy cập https://aistudio.google.com/
    2. Đăng nhập với tài khoản Google
    3. Vào "Get API Key" → "Create API Key"
    4. Copy và paste vào ô bên trái
    
    **Lưu ý:** App này sử dụng REST API trực tiếp, tương thích với mọi phiên bản Python!
    """)
else:
    # Test API key
    with st.spinner("🔍 Kiểm tra API key..."):
        if test_api_key(api_key, model_choice):
            st.success(f"✅ Đã kết nối thành công với {model_choice}")
            
            # Container cho chat
            chat_container = st.container()
            
            # Hiển thị lịch sử chat
            with chat_container:
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>👤 Bạn:</strong><br>
                            {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(f"⏰ {message['timestamp']}")
                    else:
                        model_name = message.get("model", "AI")
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>🤖 {model_name}:</strong><br>
                            <div style="white-space: pre-wrap; line-height: 1.2; margin: 0.5rem 0;">{message["content"].replace('<', '&lt;').replace('>', '&gt;')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(f"⏰ {message['timestamp']}")
                        print(message["content"])
            
            # Input cho tin nhắn mới
            st.markdown("---")
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_area(
                    "Nhập tin nhắn của bạn:",
                    height=100,
                    placeholder="Hãy hỏi tôi bất cứ điều gì...",
                    key="user_input"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                send_button = st.button("📤 Gửi", use_container_width=True)
                
            # Shortcut Enter để gửi
            if st.checkbox("Nhấn Ctrl+Enter để gửi", value=True):
                st.markdown("💡 *Tip: Checkbox này chỉ để nhắc nhở, bạn vẫn cần click nút Gửi*")
            
            # Xử lý gửi tin nhắn
            if send_button and user_input.strip():
                # Thêm tin nhắn của user
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": timestamp
                })
                
                # Hiển thị spinner khi đang xử lý
                with st.spinner(f"🤔 {model_choice} đang suy nghĩ..."):
                    response = call_gemini_api(
                        user_input, 
                        model_choice, 
                        api_key, 
                        temp=temperature, 
                        max_tok=max_tokens,
                        top_p=top_p,
                        top_k=top_k
                    )
                
                # Thêm phản hồi của AI
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "model": model_choice,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                # Clear input và refresh
                st.rerun()
                
        else:
            st.error("❌ API key không hợp lệ hoặc model không khả dụng. Vui lòng kiểm tra lại!")
            st.info("**Kiểm tra:**\n- API key có đúng không?\n- Model có được hỗ trợ không?\n- Kết nối internet ổn định không?")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    💡 <strong>Mẹo:</strong> Thử điều chỉnh các thông số để có trải nghiệm chat tốt hơn<br>
    🔧 Được xây dựng với Streamlit & Google AI Studio REST API<br>
    ✅ <strong>Tương thích với Python 3.13+</strong>
</div>
""", unsafe_allow_html=True)