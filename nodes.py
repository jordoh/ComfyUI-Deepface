import cv2
import folder_paths
import torch
import os

from deepface import DeepFace
import numpy as np
from PIL import Image

class DeepfacePrepareNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }
 
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ()
 
    FUNCTION = "run"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "deepface/prepare"
 
    def run(self, images):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            "deepface", folder_paths.get_temp_directory(), images[0].shape[1], images[0].shape[0]
        )
        output_images = []
        for image in images:
            i = 255. * image.cpu().numpy()
            image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            file = f"{filename}_{counter:05}_.png"
            full_path = os.path.join(full_output_folder, file)
            image.save(full_path, compress_level=4)

            image = image.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            detected_faces = DeepFace.extract_faces(full_path, detector_backend="retinaface")
            output_images.append(image)

            for detected_face in detected_faces:
                print(detected_face["confidence"])
                face_temp = detected_face["face"][:, :, ::-1]
                face_temp = face_temp * 255
                full_face_image_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_face.png")
                cv2.imwrite(full_face_image_path, face_temp)
                print("Detected face saved in:", full_face_image_path)

            counter += 1

        print(f"Saved {len(output_images)}")
        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
        else:
            output_image = output_images[0]

        return (output_image,)

class DeepfaceVerifyNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "image": ("IMAGE",),

            },
        }

    RETURN_TYPES = ("SCORE",)
    RETURN_NAMES = ()

    FUNCTION = "test"

    # OUTPUT_NODE = False

    CATEGORY = "deepface/verify"

    def test(self):
        return ()

NODE_CLASS_MAPPINGS = {
    "DeepfacePrepare": DeepfacePrepareNode,
    "DeepfaceVerify": DeepfaceVerifyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepfacePrepare": "Deepface Prepare",
    "DeepfaceVerify": "Deepface Verify",
}
