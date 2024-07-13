import torch

from deepface import DeepFace
import numpy as np
import os

import comfy.utils
import folder_paths

def comfy_image_from_deepface_image(deepface_image):
    image_data = np.array(deepface_image).astype(np.float32)
    return torch.from_numpy(image_data)[None,]

def deepface_image_from_comfy_image(comfy_image):
    image_data = np.clip(255 * comfy_image.cpu().numpy(), 0, 255).astype(np.uint8)
    return image_data[:, :, ::-1]  # Convert RGB to BGR

def prepare_deepface_home():
    deepface_path = os.path.join(folder_paths.models_dir, "deepface")

    # Deepface requires a specific structure within the DEEPFACE_HOME directory
    deepface_dot_path = os.path.join(deepface_path, ".deepface")
    deepface_weights_path = os.path.join(deepface_dot_path, "weights")
    if not os.path.exists(deepface_weights_path):
        os.makedirs(deepface_weights_path)

    os.environ["DEEPFACE_HOME"] = deepface_path

def result_from_images_with_measurements(images_with_measurements, sort_by):
    images_with_measurements.sort(key=lambda row: row[1] if sort_by == "distance" else (-row[2], row[1]))

    images = [row[0] for row in images_with_measurements]
    distances = [row[1] for row in images_with_measurements]
    ratios = [row[2] for row in images_with_measurements]

    if len(images) > 0:
        return torch.stack(images, dim=0), distances, ratios
    else:
        return (None, None, None,)

class DeepfaceExtractFacesNode:
    def __init__(self):
        prepare_deepface_home()
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

        progress_bar = comfy.utils.ProgressBar(len(images))

        output_images = []
        for image in images:
            progress_bar.update(1)

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
        prepare_deepface_home()
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "reference_images": ("IMAGE",),
                "distance_threshold": ("FLOAT", {
                    "default": 0.6,
                    "display": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "step": 0.01,
                }),
                "ratio_threshold": ("FLOAT", {
                    "default": 0.7,
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
                    "default": "ssd",
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
                    "default": "VGG-Face",
                }),
                "sort_by": ([
                    "distance",
                    "ratio"
                ], {
                    "default": "distance",
                })
            },
        }

    RETURN_TYPES = ("IMAGE", "NUMBER", "NUMBER", "IMAGE", "NUMBER", "NUMBER",)
    RETURN_NAMES = (
        "verified_images",
        "verified_image_distances",
        "verified_image_verified_ratios",
        "rejected_images",
        "rejected_image_distances",
        "rejected_image_verified_ratios",
    )

    FUNCTION = "run"

    CATEGORY = "deepface"

    def run(self, images, reference_images, distance_threshold, ratio_threshold, detector_backend, model_name, sort_by):
        deepface_reference_images = []
        for reference_image in reference_images:
            deepface_reference_images.append(deepface_image_from_comfy_image(reference_image))

        total_steps = len(deepface_reference_images) * len(images)
        progress_bar = comfy.utils.ProgressBar(total_steps)

        rejected_image_tuples = []
        verified_image_tuples = []
        image_counter = 0
        for image in images:
            print(f"Deepface verify { image_counter + 1 }/{ len(images) }")

            comparison_image = deepface_image_from_comfy_image(image)

            reference_image_counter = 1
            total_distance = 0
            verified_images_count = 0
            for deepface_reference_image in deepface_reference_images:
                progress_bar.update(1)

                result = None
                try:
                    result = DeepFace.verify(
                        deepface_reference_image,
                        comparison_image,
                        detector_backend=detector_backend,
                        enforce_detection=False,
                        model_name=model_name
                    )
                except ValueError as e:
                    print(f"  Reference image { reference_image_counter }/{ len(reference_images) }: ERROR: { e }")
                    continue

                distance = result["distance"]
                is_verified = result["verified"]
                print(f"  Reference image { reference_image_counter }/{ len(reference_images) }: distance={ distance } verified={ is_verified }")
                reference_image_counter += 1
                total_distance += distance
                if is_verified:
                    verified_images_count += 1

            average_distance = total_distance / len(deepface_reference_images)
            verified_ratio = round(verified_images_count / len(deepface_reference_images), 2)
            print(f"Average distance: { average_distance } ({ verified_ratio } verified)")

            if average_distance < distance_threshold and verified_ratio >= ratio_threshold:
                verified_image_tuples.append((image, average_distance, verified_ratio))
            else:
                rejected_image_tuples.append((image, average_distance, verified_ratio))

            image_counter += 1

        return result_from_images_with_measurements(verified_image_tuples, sort_by) + result_from_images_with_measurements(rejected_image_tuples, sort_by)

NODE_CLASS_MAPPINGS = {
    "DeepfaceExtractFaces": DeepfaceExtractFacesNode,
    "DeepfaceVerify": DeepfaceVerifyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepfaceExtractFaces": "Deepface Extract Faces",
    "DeepfaceVerify": "Deepface Verify",
}
