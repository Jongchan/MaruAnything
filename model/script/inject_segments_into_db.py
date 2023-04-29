import argparse
import os
import pickle

import cv2
import mysql.connector
from mysql.connector import Error
import numpy as np

data_dir = "/home/maru/model/MaruAnything/model/segments"

parser = argparse.ArgumentParser()
parser.add_argument("mysql_password", type=str, help="MySQL database password")
parser.parse_args()

def create_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name,
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

connection = create_connection("localhost", "maru", args.mysql_password, "anything")
cursor = connection.cursor()

# Create table
try:
    cursor.execute("CREATE TABLE segments ("
                   "ImageID VARCHAR(255), "
                   "SegmentID INT(255), "
                   "ContourPoints TEXT(65535))")
except:
    pass


class Polygon(object):
    def __init__(self, contour, holes=[]):
        self.contour = contour
        self.holes = holes

    def __len__(self):
        x_max = max(np.amax(self.contour[:, 0]), 1)  # in case, 0 is returned
        y_max = max(np.amax(self.contour[:, 1]), 1)
        counter_img = np.zeros((x_max, y_max), dtype=np.uint8)
        cv2.fillPoly(counter_img, [self.contour], 1)
        return counter_img.sum()

    def __repr__(self):
        output_str = (
            "["  
            + ", ".join([f"[{x}, {y}]" for x, y in self.contour])
            + "]"
        )
        if len(self.holes):
            for hole in self.holes:
                output_str += (
                    ", [" 
                    + ", ".join([f"[{x}, {y}]" for x, y in hole])
                    + "]"
                )
        return f"[{output_str}]"


# Insert entry
for image_id in sorted(os.listdir(data_dir)):
    image_dir = f"{data_dir}/{image_id}"
    for segment_fname in os.listdir(image_dir):
        with open(f"{image_dir}/{segment_fname}", "rb") as f:
            pkl_data = pickle.load(f)
        segment_idx = pkl_data["segment_idx"]
        binary_mask = pkl_data["binary_mask"]

        # Skip if already added before
        cursor.execute("SELECT * FROM segments "
                       f"WHERE ImageID='{image_id}' AND SegmentID='{segment_idx}';")
        if len(cursor.fetchall()):
            continue

        # Find contours
        mask_img = np.zeros_like(binary_mask, dtype=np.uint8)
        mask_img[binary_mask] = 255
        contours, hierarchy = cv2.findContours(
            mask_img,
            mode=cv2.RETR_CCOMP,  # two-level hierarchy (outside, then holes)
            method=cv2.CHAIN_APPROX_TC89_KCOS,  # this produces the smallest WKT
        )
        hierarchy = hierarchy[0]

        # ref. https://gist.github.com/stefano-malacrino/7d429e5d12854b9e51b187170e812fa4
        def _depth_first_search(_polygons, _contours, _hierarchy, sibling_id, is_outer,
                                siblings):
            while sibling_id != -1:
                contour = _contours[sibling_id].squeeze(axis=1)
                if len(contour) >= 3:
                    first_child_id = _hierarchy[sibling_id][2]
                    children = [] if is_outer else None
                    _depth_first_search(_polygons, _contours, _hierarchy, first_child_id,
                                        not is_outer, children)
                    if is_outer:
                        polygon = Polygon(contour, holes=children)
                        _polygons.append(polygon)
                    else:
                        siblings.append(contour)
                sibling_id = _hierarchy[sibling_id][0]

        # Convert contours to polygons
        polygons = []
        _depth_first_search(polygons, contours, hierarchy, 0, True, [])
        if len(polygons) > 1: 
            max_idx = np.argmax([len(poly) for poly in polygons])
            polygon = polygons[max_idx]
        elif len(polygons) == 0:
            continue
        else:
            polygon = polygons[0]

        # # Sanity check by drawing
        # img_path = f"/home/maru/data/VOCdevkit/VOC2012/JPEGImages/{image_id}.jpg"
        # bgr_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # seg_img = np.zeros_like(bgr_img)
        # cv2.drawContours(seg_img, [polygon.contour] + polygon.holes, 
        #                  contourIdx=-1, color=(255,), thickness=cv2.FILLED)
        # out_img = (0.5 * bgr_img + 0.5 * seg_img).astype(np.uint8)
        # out_path = f"/home/maru/tmp_wookie/{image_id}/{segment_idx}.jpg"
        # os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # cv2.imwrite(out_path, out_img)

        # Write to table
        contour_str = str(polygon)
        cursor.execute("INSERT INTO segments (ImageID, SegmentID, ContourPoints) "
                       f"VALUES ('{image_id}', {segment_idx}, '{contour_str}');")
    connection.commit()
    print(f"Wrote for image {image_id}")
