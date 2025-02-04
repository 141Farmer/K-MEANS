import cv2
import numpy as np
import mediapipe as mp

def detect_skin(image_path, output_path):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = segment.process(image_rgb)
    mask = result.segmentation_mask

   
    mask = (mask > 0.5).astype(np.uint8) * 255  

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    
    lower_hsv = np.array([0, 40, 50], dtype=np.uint8)
    upper_hsv = np.array([25, 255, 255], dtype=np.uint8)

    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)

    skin_mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    skin_mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    skin_mask = cv2.bitwise_and(skin_mask_hsv, skin_mask_ycrcb)

    
    final_mask = cv2.bitwise_and(skin_mask, mask)

   
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    
    skin = cv2.bitwise_and(image, image, mask=final_mask)

    cv2.imwrite(output_path, skin)


import os
'''
def rename_images(folder_path):

    files = os.listdir(folder_path)
    
    image_files = [f for f in files ]

    for i, filename in enumerate(image_files, start=1):
        new_name = f"real_train_{i+1}.jpg"  
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}")

rename_images("Real-Image/")
'''
images=42
for i in range(images):
          image_path=f"Real-Image/real_train_{i+2}.jpg"
          print(image_path)
          output_path=f"Mask-Image/mask_train_{i+2}.jpg"
          detect_skin(image_path,output_path)


'''
def resize_images(folder_path, new_width, new_height):
    files = os.listdir(folder_path)

    for filename in files:
        image_path = os.path.join(folder_path, filename)

        # Read the image
        image = cv2.imread(image_path)
        
        if image is not None:
            # Resize the image
            resized_image = cv2.resize(image, (new_width, new_height))

            # Overwrite the original image
            cv2.imwrite(image_path, resized_image)

            print(f"✅ Resized {filename} → {new_width}x{new_height}")
        else:
            print(f"❌ Could not read {filename}")

# Example usage
folder_path = "Real-Image/"  # Change this to your folder path
new_width, new_height = 1200, 800  # Change this to your desired size

resize_images(folder_path, new_width, new_height)
'''