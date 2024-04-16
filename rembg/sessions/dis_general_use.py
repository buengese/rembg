import os
from typing import List

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

from .base import BaseSession
from pathlib import Path


class DisSession(BaseSession):
    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        """
        Predicts the mask image for the input image.

        This method takes a PILImage object as input and returns a list of PILImage objects as output. It performs several image processing operations to generate the mask image.

        Parameters:
            img (PILImage): The input image.

        Returns:
            List[PILImage]: A list of PILImage objects representing the generated mask image.
        """
        ort_outs = self.inner_session.run(
            None,
            self.normalize(img, (0.485, 0.456, 0.406), (1.0, 1.0, 1.0), (1024, 1024)),
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
    def name(cls, *args, **kwargs):
        """
        Returns the name of the model.

        This class method returns the name of the model.

        Parameters:
            args: Additional positional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            str: The name of the model.
        """
        return "isnet-general-use"
