import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL.Image import Image as PILImage


class BaseSession:
    """This is a base class for managing a session with a machine learning model."""

    def __init__(
        self,
        model_type: str,
        sess_opts: ort.SessionOptions,
        providers=None,
        *args,
        **kwargs
    ):
        """Initialize an instance of the BaseSession class."""
        self.model_type = model_type

        self.providers = []

        _providers = ort.get_available_providers()
        if providers:
            for provider in providers:
                if provider in _providers:
                    self.providers.append(provider)
        else:
            self.providers.extend(_providers)

        self.inner_session = ort.InferenceSession(
            str(self.__class__.get_model(*args, **kwargs)),
            providers=self.providers,
            sess_options=sess_opts,
        )

    def normalize(
        self,
        img: PILImage,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        size: Tuple[int, int],
        *args,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        im = img.convert("RGB").resize(size, Image.LANCZOS)

        im_ary = np.array(im)
        im_ary = im_ary / np.max(im_ary)

        tmpImg = np.zeros((im_ary.shape[0], im_ary.shape[1], 3))
        tmpImg[:, :, 0] = (im_ary[:, :, 0] - mean[0]) / std[0]
        tmpImg[:, :, 1] = (im_ary[:, :, 1] - mean[1]) / std[1]
        tmpImg[:, :, 2] = (im_ary[:, :, 2] - mean[2]) / std[2]

        tmpImg = tmpImg.transpose((2, 0, 1))

        return {
            self.inner_session.get_inputs()[0]
            .name: np.expand_dims(tmpImg, 0)
            .astype(np.float32)
        }

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        raise NotImplementedError

    @classmethod
    def rembg_home(cls, *args, **kwargs):
        return os.path.expanduser(
            os.getenv(
                "REMBG_HOME", os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".rembg")
            )
        )

    @classmethod
    def get_model(cls, *args, **kwargs):
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
        raise NotImplementedError
