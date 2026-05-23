"""Visual prototype definitions for unified KG construction."""

# Visual prototype library: common visual patterns based on shape, category, and attributes
VISUAL_PROTOTYPES = [
    # === Basic shapes ===
    "round object",
    "spherical object",
    "ball-like object",
    "cylindrical object",
    "tube-like object",
    "rod-like object",
    "box-like object",
    "rectangular object",
    "cube-like object",
    "flat object",
    "disc-like object",
    "plate-like object",
    "elongated object",
    "stick-like object",
    "cone-shaped object",

    # === Animal categories (coarse-grained) ===
    "animal",
    "four-legged animal",
    "mammal",
    "bird",
    "flying bird",
    "water bird",
    "fish",
    "aquatic animal",
    "reptile",
    "insect",
    "furry animal",
    "large animal",
    "small animal",

    # === Plants ===
    "plant",
    "tree",
    "flower",
    "leafy plant",
    "green plant",
    "fruit",
    "vegetable",

    # === Artifacts (coarse-grained) ===
    "vehicle",
    "wheeled vehicle",
    "car-like vehicle",
    "flying vehicle",
    "water vehicle",
    "boat-like object",

    "tool",
    "handheld tool",
    "cutting tool",
    "pointed tool",

    "container",
    "bottle-like container",
    "cup-like container",
    "box-like container",

    "furniture",
    "seating furniture",
    "table-like furniture",

    "electronic device",
    "screen device",
    "handheld device",

    "musical instrument",
    "string instrument",
    "wind instrument",
    "percussion instrument",
    "keyboard instrument",

    "clothing",
    "headwear",
    "footwear",
    "upper body clothing",
    "lower body clothing",

    "food item",
    "prepared food",
    "baked food",
    "meat food",

    "sports equipment",
    "ball sports equipment",
    "racket sports equipment",

    "building structure",
    "indoor structure",
    "outdoor structure",

    # === Visual attributes ===
    "metallic object",
    "wooden object",
    "plastic object",
    "glass object",
    "fabric object",

    "transparent object",
    "shiny object",
    "matte object",
    "textured object",
    "smooth object",

    "colorful object",
    "monochrome object",
    "striped pattern",
    "spotted pattern",
    "patterned object",

    "soft object",
    "hard object",
    "flexible object",
    "rigid object",

    # === Functional properties (visually inferable) ===
    "portable object",
    "wearable object",
    "edible object",
    "decorative object",
    "functional tool",
]

# Categorize prototypes for better organization and potential category-based retrieval
PROTOTYPE_CATEGORIES = {
    "shape": [
        "round object", "spherical object", "ball-like object",
        "cylindrical object", "tube-like object", "rod-like object",
        "box-like object", "rectangular object", "cube-like object",
        "flat object", "disc-like object", "plate-like object",
        "elongated object", "stick-like object", "cone-shaped object",
    ],
    "animal": [
        "animal", "four-legged animal", "mammal", "bird",
        "flying bird", "water bird", "fish", "aquatic animal",
        "reptile", "insect", "furry animal", "large animal", "small animal",
    ],
    "plant": [
        "plant", "tree", "flower", "leafy plant", "green plant",
        "fruit", "vegetable",
    ],
    "artifact": [
        "vehicle", "tool", "container", "furniture", "electronic device",
        "musical instrument", "clothing", "food item", "sports equipment",
        "building structure",
    ],
    "material": [
        "metallic object", "wooden object", "plastic object",
        "glass object", "fabric object",
    ],
    "visual_attribute": [
        "transparent object", "shiny object", "matte object",
        "textured object", "smooth object", "colorful object",
        "monochrome object", "striped pattern", "spotted pattern",
        "patterned object",
    ],
    "physical_property": [
        "soft object", "hard object", "flexible object", "rigid object",
        "portable object", "wearable object", "edible object",
        "decorative object", "functional tool",
    ],
}

def get_all_prototypes():
    """Get all visual prototypes."""
    return VISUAL_PROTOTYPES

def get_prototype_category(prototype):
    """Get category of a prototype."""
    for category, prototypes in PROTOTYPE_CATEGORIES.items():
        if prototype in prototypes:
            return category
    return "other"
