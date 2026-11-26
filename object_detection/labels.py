"""COCO label groups used by the object counting page.

The MediaPipe EfficientDet models are trained on COCO (80 classes).
We split the classes we care about into three groups that the UI can toggle:
person, vehicle and everyday objects. Other classes (animals, traffic lights,
...) are ignored.
"""

PERSON = "person"
VEHICLE = "vehicle"
EVERYDAY = "everyday"

GROUPS = (PERSON, VEHICLE, EVERYDAY)

PERSON_LABELS = {"person"}

VEHICLE_LABELS = {"bicycle", "car", "motorcycle", "bus", "truck", "train"}

EVERYDAY_LABELS = {
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "sports ball", "frisbee", "skateboard", "tennis racket", "baseball bat",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
}

# Vietnamese display names for the web UI (OpenCV cannot draw diacritics,
# so the video overlay keeps the English COCO names).
LABEL_VI = {
    "person": "Người",
    "bicycle": "Xe đạp",
    "car": "Ô tô",
    "motorcycle": "Xe máy",
    "bus": "Xe buýt",
    "truck": "Xe tải",
    "train": "Tàu hỏa",
    "backpack": "Ba lô",
    "umbrella": "Ô (dù)",
    "handbag": "Túi xách",
    "tie": "Cà vạt",
    "suitcase": "Vali",
    "sports ball": "Quả bóng",
    "frisbee": "Đĩa ném",
    "skateboard": "Ván trượt",
    "tennis racket": "Vợt tennis",
    "baseball bat": "Gậy bóng chày",
    "bottle": "Chai",
    "wine glass": "Ly rượu",
    "cup": "Cốc",
    "fork": "Nĩa",
    "knife": "Dao",
    "spoon": "Thìa",
    "bowl": "Bát",
    "banana": "Chuối",
    "apple": "Táo",
    "sandwich": "Bánh mì kẹp",
    "orange": "Cam",
    "broccoli": "Súp lơ xanh",
    "carrot": "Cà rốt",
    "hot dog": "Xúc xích kẹp",
    "pizza": "Pizza",
    "donut": "Bánh donut",
    "cake": "Bánh ngọt",
    "chair": "Ghế",
    "couch": "Ghế sofa",
    "potted plant": "Chậu cây",
    "bed": "Giường",
    "dining table": "Bàn ăn",
    "toilet": "Bồn cầu",
    "tv": "Tivi",
    "laptop": "Laptop",
    "mouse": "Chuột máy tính",
    "remote": "Điều khiển",
    "keyboard": "Bàn phím",
    "cell phone": "Điện thoại",
    "microwave": "Lò vi sóng",
    "oven": "Lò nướng",
    "toaster": "Máy nướng bánh",
    "sink": "Bồn rửa",
    "refrigerator": "Tủ lạnh",
    "book": "Sách",
    "clock": "Đồng hồ",
    "vase": "Bình hoa",
    "scissors": "Kéo",
    "teddy bear": "Gấu bông",
    "hair drier": "Máy sấy tóc",
    "toothbrush": "Bàn chải đánh răng",
}


def group_of(label):
    """Return the group of a COCO label, or None if the label is ignored."""
    if label in PERSON_LABELS:
        return PERSON
    if label in VEHICLE_LABELS:
        return VEHICLE
    if label in EVERYDAY_LABELS:
        return EVERYDAY
    return None


def label_vi(label):
    return LABEL_VI.get(label, label)
