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
    RETURN_NAMES = ("face_images",)
 
    FUNCTION = "run"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "deepface/prepare"
 
    def run(self, images):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            "deepface", folder_paths.get_temp_directory(), images[0].shape[1], images[0].shape[0]
        )
        target_face_size = (224, 224)
        output_images = []
        for image in images:
            i = 255. * image.cpu().numpy()
            image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            file = f"{filename}_{counter:05}_.png"
            full_path = os.path.join(full_output_folder, file)
            image.save(full_path, compress_level=4)

            detected_faces = DeepFace.extract_faces(full_path, detector_backend="retinaface", target_size=target_face_size)

            for detected_face in detected_faces:
                # print(detected_face["confidence"])
                face_image = np.array(detected_face["face"]).astype(np.float32)
                face_image = torch.from_numpy(face_image)[None,]
                output_images.append(face_image)

            counter += 1

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
