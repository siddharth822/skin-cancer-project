"""Conservative photo-quality gate, not a trained skin/lesion detector."""
import numpy as np
from PIL import Image


def validate_photo(image: Image.Image):
    """Reject obvious unusable inputs before the six-class classifier runs."""
    if min(image.size) < 96:
        raise ValueError('Image is too small. Upload an original, clear close-up skin photo at least 96 pixels wide and high.')
    sample = image.convert('RGB').copy()
    sample.thumbnail((256, 256))
    rgb = np.asarray(sample)
    gray = rgb.astype(np.float32).mean(axis=2)
    if float(gray.std()) < 3 or float((gray < 8).mean()) > .95 or float((gray > 247).mean()) > .95:
        raise ValueError('Image is blank or poorly exposed. Take a clear skin photo in natural light.')
    # Flat-color graphics and initials avatars have few colors or large uniform areas.
    # This is independent of skin tone and never uses classifier confidence.
    colors, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
    if len(colors) < 64 or float(counts.max() / counts.sum()) > .50:
        raise ValueError('This looks like an icon, text, or a graphic rather than a usable photo. Upload a clear close-up photo of the skin lesion.')
    if np.percentile(rgb.max(axis=2).astype(float) - rgb.min(axis=2), 95) < 4:
        raise ValueError('Upload an original color skin photo; grayscale graphics and scans cannot be screened reliably.')
