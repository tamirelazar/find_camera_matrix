import sys
import os

def detect_corners(): pass


if __name__ == "__main__":
    try:
        images_path = sys.argv[1]
        if not os.path.isdir(images_path):
            raise Exception("Please create directory data/ and place images inside.")
    except Exception:
        raise Exception("GIVE ME IMAGES!!")

    images = os.listdir(images_path)
    real_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    image_coords =