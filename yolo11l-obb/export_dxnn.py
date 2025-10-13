# IMPORTANT: Import this BEFORE any ultralytics imports
import custom_ultralytics_setup

import os
import ssl
import urllib3
from ultralytics import YOLO

def main():
    try:
        # Disable SSL verification (to resolve SSL issues due to corporate firewalls, etc.)
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Set offline mode
        os.environ['YOLO_OFFLINE'] = '1'
        os.environ['ULTRALYTICS_OFFLINE'] = '1'

        # Load YOLOv11l-obb model
        model = YOLO('models/yolo11l-obb.pt')

        # Convert model to DXNN format (opset=12 specified for general compatibility)
        model.export(format='deepx')

        print("Successfully converted YOLOv11l-obb model to DXNN format. 'models/yolo11l-obb.dxnn' file has been created.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
