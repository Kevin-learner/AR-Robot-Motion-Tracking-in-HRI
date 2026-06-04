import requests
import io
import numpy as np

# 发送一个极其简单的测试包
test_data = {'xyz': np.array([[0,0,0]], dtype=np.float32), 'K': np.eye(3)}
memfile = io.BytesIO()
np.save(memfile, np.array([test_data], dtype=object))
memfile.seek(0)

# 发送
url = "http://100.116.99.44:5000/predict_grasp"
try:
    r = requests.post(url, data=memfile.read())
    print(f"状态码: {r.status_code}, 响应内容: {r.text}")
except Exception as e:
    print(f"发送失败: {e}")