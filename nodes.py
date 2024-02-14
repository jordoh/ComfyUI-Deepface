import torch

from deepface import DeepFace
import numpy as np

def comfy_image_from_deepface_image(deepface_image):
    image_data = np.array(deepface_image).astype(np.float32)
    return torch.from_numpy(image_data)[None,]

def deepface_image_from_comfy_image(comfy_image):
    image_data = np.clip(255 * comfy_image.cpu().numpy(), 0, 255).astype(np.uint8)
    return image_data[:, :, ::-1]  # Convert RGB to BGR

class DeepfaceExtractFacesNode:
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
 
    CATEGORY = "deepface"
 
    def run(self, images):
        target_face_size = (224, 224)

        output_images = []
        for image in images:
            image = deepface_image_from_comfy_image(image)

            detected_faces = DeepFace.extract_faces(
                image,
                detector_backend="retinaface",
                enforce_detection=False,
                target_size=target_face_size,
            )

            for detected_face in detected_faces:
                # print(detected_face["confidence"])
                face_image = comfy_image_from_deepface_image(detected_face["face"])
                output_images.append(face_image)

        if len(output_images) > 0:
            return (torch.cat(output_images, dim=0),)
        else:
            return ((),)

class DeepfaceVerifyNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "reference_images": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.3,
                    "display": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "step": 0.01,
                }),
                "detector_backend": ([
                     "opencv",
                     "ssd",
                     "dlib",
                     "mtcnn",
                     "retinaface",
                     "mediapipe",
                     "yolov8",
                     "yunet",
                     "fastmtcnn",
                ], {
                    "default": "retinaface",
                }),
                "model_name": ([
                     "VGG-Face",
                     "Facenet",
                     "Facenet512",
                     "OpenFace",
                     "DeepFace",
                     "DeepID",
                     "ArcFace",
                     "Dlib",
                     "SFace",
                ], {
                    "default": "Facenet512",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("verified_images",)

    FUNCTION = "run"

    CATEGORY = "deepface"

    def run(self, images, reference_images, threshold, detector_backend, model_name):
        deepface_reference_images = []
        for reference_image in reference_images:
            deepface_reference_images.append(deepface_image_from_comfy_image(reference_image))

        output_images_with_distances = []
        for image in images:
            print("Deepface verify")

            comparison_image = deepface_image_from_comfy_image(image)

            reference_image_counter = 1
            total_distance = 0
            for deepface_reference_image in deepface_reference_images:
                result = DeepFace.verify(
                    deepface_reference_image,
                    comparison_image,
                    detector_backend=detector_backend,
                    enforce_detection=False,
                    model_name=model_name
                )
                distance = result["distance"]
                print(f"  Distance to face image #{reference_image_counter}: {distance} ({result['verified']})")
                reference_image_counter += 1
                total_distance += distance

            average_distance = total_distance / len(deepface_reference_images)
            print(f"Average distance: {average_distance}")
            if average_distance <= threshold:
                output_images_with_distances.append((image, average_distance))

        output_images_with_distances.sort(key=lambda row: row[1])
        output_images = [row[0] for row in output_images_with_distances]

        if len(output_images) > 0:
            return (torch.stack(output_images, dim=0),)
        else:
            return ((),)

NODE_CLASS_MAPPINGS = {
    "DeepfaceExtractFaces": DeepfaceExtractFacesNode,
    "DeepfaceVerify": DeepfaceVerifyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepfaceExtractFaces": "Deepface Extract Faces",
    "DeepfaceVerify": "Deepface Verify",
}
