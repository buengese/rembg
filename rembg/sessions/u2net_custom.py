import os
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL.Image import Image as PILImage

from .base import BaseSession


class U2netCustomSession(BaseSession):
    """This is a class representing a custom session for the U2net model."""

    def __init__(
        self,
        model_type: str,
        sess_opts: ort.SessionOptions,
        providers=None,
        *args,
        **kwargs
    ):
        """
        Initialize a new U2netCustomSession object.

        Parameters:
            model_type (str): The name of the model.
            sess_opts (ort.SessionOptions): The session options.
            providers: The providers.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If model_path is None.
        """
        model_path = kwargs.get("model_path")
        if model_path is None:
            raise ValueError("model_path is required")

        super().__init__(model_type, sess_opts, providers, *args, **kwargs)

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        """
        Predict the segmentation mask for the input image.

        Parameters:
            img (PILImage): The input image.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List[PILImage]: A list of PILImage objects representing the segmentation mask.
        """
        ort_outs = self.inner_session.run(
            None,
            self.normalize(
                img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (320, 320)
            ),
        )

        pred = ort_outs[0][:, 0, :, :]

        ma = np.max(pred)
        mi = np.min(pred)

        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)

        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        mask = mask.resize(img.size, Image.LANCZOS)

        return [mask]

    @classmethod
    def get_model(cls, *args, **kwargs):
        model_path = kwargs.get("model_path")
        if model_path is not None:
            return os.path.abspath(os.path.expanduser(model_path))

        fname = f"{cls.name(*args, **kwargs)}.onnx"
        path = Path(cls.rembg_home(*args, **kwargs)).joinpath(fname)
        if path.exists():
            return path

        path = Path(os.path.dirname(__file__)).parent.joinpath("models", fname)
        if path.exists():
            return path

        raise FileNotFoundError(f"Model file {path} not found.")


    @classmethod
    def name(cls, *args, **kwargs):
        """
        Get the name of the model.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the model.
        """
        return "u2net_custom"
