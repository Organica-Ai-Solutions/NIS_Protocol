import base64, io
from PIL import Image
import numpy as np
arr = np.zeros((480, 848, 3), dtype=np.uint8)
arr[:, :, :] = [80, 100, 120]
buf = io.BytesIO()
Image.fromarray(arr).save(buf, format="JPEG", quality=85)
print(base64.b64encode(buf.getvalue()).decode())
