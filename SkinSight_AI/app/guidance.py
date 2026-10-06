GUIDANCE = {
    "ACK": {
        "name": "Actinic Keratosis",
        "risk": "Needs dermatologist review",
        "message": "Often related to sun damage and can be precancerous. Arrange a dermatologist review.",
        "urgency": "Book a routine-to-soon dermatology appointment, especially if changing, tender, crusting, or bleeding."
    },
    "BCC": {
        "name": "Basal Cell Carcinoma",
        "risk": "High concern",
        "message": "This class represents a skin cancer category in the training dataset.",
        "urgency": "Arrange prompt in-person dermatology assessment and biopsy consideration."
    },
    "MEL": {
        "name": "Melanoma",
        "risk": "High concern",
        "message": "This class represents melanoma, which requires professional assessment.",
        "urgency": "Arrange prompt dermatologist evaluation. Do not rely on this image result to confirm or rule out melanoma."
    },
    "NEV": {
        "name": "Melanocytic Nevus",
        "risk": "Usually lower concern",
        "message": "Many nevi are benign, but changing or unusual lesions still need examination.",
        "urgency": "Seek review if the lesion is new, changing, asymmetric, bleeding, itching, painful, or looks different from your other moles."
    },
    "SCC": {
        "name": "Squamous Cell Carcinoma",
        "risk": "High concern",
        "message": "This class represents a skin cancer category in the training dataset.",
        "urgency": "Arrange prompt in-person dermatology assessment and biopsy consideration."
    },
    "SEK": {
        "name": "Seborrheic Keratosis",
        "risk": "Usually lower concern",
        "message": "This is commonly benign, but visual similarity with other lesions can occur.",
        "urgency": "Seek clinical review if rapidly changing, bleeding, painful, or uncertain."
    },
}

def stage_info(label: str):
    if label in {"BCC", "MEL", "SCC"}:
        return {
            "status": "Not determinable from a photo",
            "explanation": (
                "Clinical cancer Stage I–IV cannot be assigned safely from this image alone. "
                "Staging requires pathology/biopsy and may require tumor depth, ulceration, "
                "lymph-node findings, imaging, or evidence of spread."
            ),
        }
    return {
        "status": "Not applicable from image classification",
        "explanation": (
            "The image model provides lesion-class screening, not clinical cancer staging."
        ),
    }
