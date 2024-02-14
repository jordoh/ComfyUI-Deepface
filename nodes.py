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
        target_face_size = (224, 224)

        output_images = []
        for image in images:
            image_data = np.clip(255 * image.cpu().numpy(), 0, 255).astype(np.uint8)
            image_data = image_data[:, :, ::-1]  # Convert RGB to BGR

            detected_faces = DeepFace.extract_faces(image_data, detector_backend="retinaface", enforce_detection=False, target_size=target_face_size)

            for detected_face in detected_faces:
                # print(detected_face["confidence"])
                face_image = np.array(detected_face["face"]).astype(np.float32)
                face_image = torch.from_numpy(face_image)[None,]
                output_images.append(face_image)

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
                "images": ("IMAGE",),
                "face_images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("verified_images",)

    FUNCTION = "run"

    # OUTPUT_NODE = False

    CATEGORY = "deepface/verify"

    def run(self, images, face_images):
        model_name = "Facenet512"

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            "deepface-verify", folder_paths.get_temp_directory(), images[0].shape[1], images[0].shape[0]
        )

        face_image_paths = []
        for face_image in face_images:
            face_image = 255. * face_image.cpu().numpy()
            face_image = Image.fromarray(np.clip(face_image, 0, 255).astype(np.uint8))

            face_image_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.png")
            face_image.save(face_image_path, compress_level=4)
            face_image_paths.append(face_image_path)

            counter += 1

        verified_images = []
        for image in images:
            comparison_image = 255. * image.cpu().numpy()
            comparison_image = Image.fromarray(np.clip(comparison_image, 0, 255).astype(np.uint8))

            image_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.png")
            comparison_image.save(image_path, compress_level=4)

            avg_distance = 0

            is_verified = False
            org_img_counter = 1
            for face_image_path in face_image_paths:
                result = DeepFace.verify(face_image_path, image_path, detector_backend="retinaface", model_name=model_name)
                distance_score = result["distance"]

                print(f"Distance of AI image #{counter} to org image #{org_img_counter}: {distance_score} ({result['verified']})")
                avg_distance = avg_distance + distance_score
                org_img_counter = org_img_counter + 1

                if result['verified']:
                    is_verified = True
            #     distance_score = avg_distance / len(org_images)
            #     distance_scores.append((generated_image_file, distance_score))
            #     print(f"AI image {counter} / {len(image_files)} - {distance_score}")

            if is_verified:
                verified_images.append(image)

            counter += 1

        if len(verified_images) > 1:
            output_image = torch.cat(verified_images, dim=0)
        else:
            output_image = verified_images[0]

        return (output_image,)

NODE_CLASS_MAPPINGS = {
    "DeepfacePrepare": DeepfacePrepareNode,
    "DeepfaceVerify": DeepfaceVerifyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepfacePrepare": "Deepface Prepare",
    "DeepfaceVerify": "Deepface Verify",
}
