import logging
import os
from copy import deepcopy
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import pooch
from jsonschema import validate
from PIL import Image
from PIL.Image import Image as PILImage

from .base import BaseSession


class SamSession(BaseSession):
    """
    This class represents a session for the Sam model.

    Args:
        model_name (str): The name of the model.
        sess_opts (ort.SessionOptions): The session options.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, model_name: str, sess_opts: ort.SessionOptions, *args, **kwargs):
        """
        Initialize a new SamSession with the given model name and session options.

        Args:
            model_name (str): The name of the model.
            sess_opts (ort.SessionOptions): The session options.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.model_name = "sam_vit_h_4b8939"

        self.target_size = 1024
        self.input_size = (684, 1024)

        providers = ort.get_available_providers()
        # Pop TensorRT Runtime due to crashing issues
        # TODO: Add back when TensorRT backend is stable
        providers = [p for p in providers if p != "TensorrtExecutionProvider"]

        if providers:
            logging.info(
                "Available providers for ONNXRuntime: %s", ", ".join(providers)
            )
        else:
            logging.warning("No available providers for ONNXRuntime")

        paths = (
            # relative to current file
            os.path.join(os.path.join(os.path.dirname(__file__), os.pardir), "models", f"{self.model_name}.encoder.onnx"),
            os.path.join(os.path.join(os.path.dirname(__file__), os.pardir), "models", f"{self.model_name}.decoder.onnx"),
        )

        # paths = self.__class__.download_models(*args, **kwargs)
        self.encoder = ort.InferenceSession(
            str(paths[0]),
            providers=providers,
            sess_options=sess_opts,
        )
        self.decoder = ort.InferenceSession(
            str(paths[1]),
            providers=providers,
            sess_options=sess_opts,
        )
        self.encoder_input_name = self.encoder.get_inputs()[0].name

    @staticmethod
    def get_input_points(prompt):
        """Get input points"""
        points = []
        labels = []
        for mark in prompt:
            if mark["type"] == "point":
                points.append(mark["data"])
                labels.append(mark["label"])
            elif mark["type"] == "rectangle":
                points.append([mark["data"][0], mark["data"][1]])  # top left
                points.append(
                    [mark["data"][2], mark["data"][3]]
                )  # bottom right
                labels.append(2)
                labels.append(3)
        points, labels = np.array(points), np.array(labels)
        return points, labels

    def run_encoder(self, encoder_inputs):
        """Run encoder"""
        output = self.encoder.run(None, encoder_inputs)
        image_embedding = output[0]
        return image_embedding

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def apply_coords(self, coords: np.ndarray, original_size, target_length):
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def normalize(
        self,
        img: np.ndarray,
        mean=(),
        std=(),
        size=(),
        *args,
        **kwargs,
    ):
        """
        Normalize the input image by subtracting the mean and dividing by the standard deviation.

        Args:
            img (np.ndarray): The input image.
            mean (tuple, optional): The mean values for normalization. Defaults to ().
            std (tuple, optional): The standard deviation values for normalization. Defaults to ().
            size (tuple, optional): The target size of the image. Defaults to ().
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            np.ndarray: The normalized image.
        """
        return img

    def run_decoder(
        self, image_embedding, original_size, transform_matrix, prompt
    ):
        """Run decoder"""
        input_points, input_labels = self.get_input_points(prompt)

        # Add a batch index, concatenate a padding point, and transform.
        onnx_coord = np.concatenate(
            [input_points, np.array([[0.0, 0.0]])], axis=0
        )[None, :, :]
        onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[
                     None, :
                     ].astype(np.float32)
        onnx_coord = self.apply_coords(
            onnx_coord, self.input_size, self.target_size
        ).astype(np.float32)

        # Apply the transformation matrix to the coordinates.
        onnx_coord = np.concatenate(
            [
                onnx_coord,
                np.ones((1, onnx_coord.shape[1], 1), dtype=np.float32),
            ],
            axis=2,
        )
        onnx_coord = np.matmul(onnx_coord, transform_matrix.T)
        onnx_coord = onnx_coord[:, :, :2].astype(np.float32)

        # Create an empty mask input and an indicator for no mask.
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        decoder_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(self.input_size, dtype=np.float32),
        }
        masks, _, _ = self.decoder.run(None, decoder_inputs)

        print(f"Shape masks: {masks.shape}")

        # Transform the masks back to the original image size.
        inv_transform_matrix = np.linalg.inv(transform_matrix)
        transformed_masks = self.transform_masks(
            masks, original_size, inv_transform_matrix
        )

        return transformed_masks

    def transform_masks(self, masks, original_size, transform_matrix):
        """Transform masks
        Transform the masks back to the original image size.
        """
        output_masks = []
        for batch in range(masks.shape[0]):
            batch_masks = []
            for mask_id in range(masks.shape[1]):
                mask = masks[batch, mask_id]
                mask = cv2.warpAffine(
                    mask,
                    transform_matrix[:2],
                    (original_size[1], original_size[0]),
                    flags=cv2.INTER_LINEAR,
                )
                batch_masks.append(mask)
            output_masks.append(batch_masks)
        return np.array(output_masks)

    def encode(self, cv_image):
        """
        Calculate embedding and metadata for a single image.
        """
        original_size = cv_image.shape[:2]

        # Calculate a transformation matrix to convert to self.input_size
        scale_x = self.input_size[1] / cv_image.shape[1]
        scale_y = self.input_size[0] / cv_image.shape[0]
        scale = min(scale_x, scale_y)
        transform_matrix = np.array(
            [
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, 1],
            ]
        )
        cv_image = cv2.warpAffine(
            cv_image,
            transform_matrix[:2],
            (self.input_size[1], self.input_size[0]),
            flags=cv2.INTER_LINEAR,
        )

        encoder_inputs = {
            self.encoder_input_name: cv_image.astype(np.float32),
        }
        image_embedding = self.run_encoder(encoder_inputs)
        return {
            "image_embedding": image_embedding,
            "original_size": original_size,
            "transform_matrix": transform_matrix,
        }

    def predict_masks(self, embedding, prompt):
        """
        Predict masks for a single image.
        """
        masks = self.run_decoder(
            embedding["image_embedding"],
            embedding["original_size"],
            embedding["transform_matrix"],
            prompt,
        )

        return masks

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        """
        Predict masks for an input image.

        This function takes an image as input and performs various preprocessing steps on the image. It then runs the image through an encoder to obtain an image embedding. The function also takes input labels and points as additional arguments. It concatenates the input points and labels with padding and transforms them. It creates an empty mask input and an indicator for no mask. The function then passes the image embedding, point coordinates, point labels, mask input, and has mask input to a decoder. The decoder generates masks based on the input and returns them as a list of images.

        Parameters:
            img (PILImage): The input image.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List[PILImage]: A list of masks generated by the decoder.
        """
        prompt = kwargs.get("sam_prompt", "{}")

        img = img.convert("RGB")
        cv_image = np.array(img)

        embedding = self.encode(cv_image)
        masks = self.predict_masks(embedding, prompt)

        # Save the masks as a single image.
        mask = np.zeros((masks.shape[2], masks.shape[3], 3), dtype=np.uint8)
        for m in masks[0, :, :, :]:
            mask[m > 0.0] = [255, 255, 255]
        mask = Image.fromarray(mask).convert("L")
        return [mask]


    @classmethod
    def name(cls, *args, **kwargs):
        """
        Class method to return a string value.

        This method returns the string value 'sam'.

        Parameters:
            cls: The class object.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The string value 'sam'.
        """
        return "sam"
