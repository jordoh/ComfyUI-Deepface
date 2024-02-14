import torch

from deepface import DeepFace
import numpy as np

def comfy_image_from_deepface_image(deepface_image):
    image_data = np.array(deepface_image).astype(np.float32)
    return torch.from_numpy(image_data)[None,]

def deepface_image_from_comfy_image(comfy_image):
    image_data = np.clip(255 * comfy_image.cpu().numpy(), 0, 255).astype(np.uint8)
    return image_data[:, :, ::-1]  # Convert RGB to BGR

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
            image = deepface_image_from_comfy_image(image)

            detected_faces = DeepFace.extract_faces(image, detector_backend="retinaface", enforce_detection=False, target_size=target_face_size)

            for detected_face in detected_faces:
                # print(detected_face["confidence"])
                face_image = comfy_image_from_deepface_image(detected_face["face"])
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
                "threshold": ("FLOAT", {
                    "default": 0.3,
                    "display": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("verified_images",)

    FUNCTION = "run"

    # OUTPUT_NODE = False

    CATEGORY = "deepface/verify"

    def run(self, images, face_images, threshold):
        detector_backend = "retinaface"
        model_name = "Facenet512"

        deepface_face_images = []
        for face_image in face_images:
            deepface_face_images.append(deepface_image_from_comfy_image(face_image))

        output_images_with_distances = []
        for image in images:
            print("Deepface verify")

            comparison_image = deepface_image_from_comfy_image(image)

            face_image_counter = 1
            total_distance = 0
            for deepface_face_image in deepface_face_images:
                result = DeepFace.verify(deepface_face_image, comparison_image, detector_backend=detector_backend, model_name=model_name)
                distance = result["distance"]

                print(f"  Distance to face image #{face_image_counter}: {distance} ({result['verified']})")
                face_image_counter += 1
                total_distance += distance

            average_distance = total_distance / len(deepface_face_images)
            print(f"Average distance: {average_distance}")
            if average_distance <= threshold:
                output_images_with_distances.append(image)

        return (torch.stack(output_images_with_distances, dim=0),)

NODE_CLASS_MAPPINGS = {
    "DeepfacePrepare": DeepfacePrepareNode,
    "DeepfaceVerify": DeepfaceVerifyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepfacePrepare": "Deepface Prepare",
    "DeepfaceVerify": "Deepface Verify",
}
