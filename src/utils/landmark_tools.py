# from mediapipe.solution import 
import math
import cv2
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial import Delaunay
import numpy as np


CONNECTIONS = {
    "lips": [
        (61, 146),
        (146, 91),
        (91, 181),
        (181, 84),
        (84, 17),
        (17, 314),
        (314, 405),
        (405, 321),
        (321, 375),
        (375, 291),
        (61, 185),
        (185, 40),
        (40, 39),
        (39, 37),
        (37, 0),
        (0, 267),
        (267, 269),
        (269, 270),
        (270, 409),
        (409, 291),
        (78, 95),
        (95, 88),
        (88, 178),
        (178, 87),
        (87, 14),
        (14, 317),
        (317, 402),
        (402, 318),
        (318, 324),
        (324, 308),
        (78, 191),
        (191, 80),
        (80, 81),
        (81, 82),
        (82, 13),
        (13, 312),
        (312, 311),
        (311, 310),
        (310, 415),
        (415, 308),
    ],

    "left eye": [
        (263, 249),
        (249, 390),
        (390, 373),
        (373, 374),
        (374, 380),
        (380, 381),
        (381, 382),
        (382, 362),
        (263, 466),
        (466, 388),
        (388, 387),
        (387, 386),
        (386, 385),
        (385, 384),
        (384, 398),
        (398, 362),
    ],


    "left eyebrow" : [
        (276, 283),
        (283, 282),
        (282, 295),
        (295, 285),
        (300, 293),
        (293, 334),
        (334, 296),
        (296, 336),
    ],

    "right eye" : [
        (33, 7),
        (7, 163),
        (163, 144),
        (144, 145),
        (145, 153),
        (153, 154),
        (154, 155),
        (155, 133),
        (33, 246),
        (246, 161),
        (161, 160),
        (160, 159),
        (159, 158),
        (158, 157),
        (157, 173),
        (173, 133),
    ],

    "right eyebrow" : [
        (46, 53),
        (53, 52),
        (52, 65),
        (65, 55),
        (70, 63),
        (63, 105),
        (105, 66),
        (66, 107),
    ],

    "oval" : [
        (10, 338),
        (338, 297),
        (297, 332),
        (332, 284),
        (284, 251),
        (251, 389),
        (389, 356),
        (356, 454),
        (454, 323),
        (323, 361),
        (361, 288),
        (288, 397),
        (397, 365),
        (365, 379),
        (379, 378),
        (378, 400),
        (400, 377),
        (377, 152),
        (152, 148),
        (148, 176),
        (176, 149),
        (149, 150),
        (150, 136),
        (136, 172),
        (172, 58),
        (58, 132),
        (132, 93),
        (93, 234),
        (234, 127),
        (127, 162),
        (162, 21),
        (21, 54),
        (54, 103),
        (103, 67),
        (67, 109),
        (109, 10)
    ],
}



def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int, image_height: int):
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def annotate_landmark(img, landmark_list, connections=True):
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils 
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    if connections is True:
        connections = mp_face_mesh.FACE_CONNECTIONS

    annotated_image = img.copy()
    image_rows, image_cols, _ = annotated_image.shape 
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list):
        landmark_px = _normalized_to_pixel_coordinates(landmark["X"], landmark["Y"],
                                                       image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px

    if connections:
        num_landmarks = len(landmark_list)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                cv2.line(annotated_image, idx_to_coordinates[start_idx],
                         idx_to_coordinates[end_idx], drawing_spec.color,
                         drawing_spec.thickness)

    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.
    for landmark_px in idx_to_coordinates.values():
        cv2.circle(annotated_image, landmark_px, drawing_spec.circle_radius,
               drawing_spec.color, drawing_spec.thickness)
        
    return annotated_image

#------------------------------------------------------

def landmark_to_array(landmark_list):
    lm_array = np.zeros([len(landmark_list), 3])
    for i, l in enumerate(landmark_list):
        lm_array[i, 0] = l["X"]
        lm_array[i, 1] = l["Y"]
        lm_array[i, 2] = l["Z"]
    
    return lm_array

def normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image):
    """Converts normalized value pair to pixel coordinates."""

    image_height, image_width = image.shape[:2]
    x_px = math.floor(normalized_x * image_width)
    y_px = math.floor(normalized_y * image_height)
    return np.array([x_px, y_px])


def connection_to_index(list_of_tuple):
    L = [list_of_tuple[0][0]] + [ t[1] for t in list_of_tuple ]
    return L
    # L = []
    # for t in list_of_tuple:
        # for z in t:
            # L.append(z)
            
    # L = list(set(L))
    # return L

def compute_contour(ptr_array):
    """
    find the convex hull surrounding the points
    return the convex hull point indices
    """
    hull = ConvexHull(ptr_array)
    contour_index = hull.vertices.tolist() # indices are ordered
    # contour_index = hull.simplices.flatten()
    # contour_index = list(set(contour_index))
    return contour_index

def compute_triangle_area(a,b,c):
    """
    give the three point a, b, c of the triangle
    compute area of it
    """
    ab = np.sqrt( ((a-b)**2).sum() )
    ac = np.sqrt( ((a-c)**2).sum() )
    bc = np.sqrt( ((b-c)**2).sum() )
    
    s = (ab+ac+bc)/2
    area = np.sqrt(s*(s-ab)*(s-bc)*(s-ac))
    
    return area

def generate_triangle(ptr_array, connection_dict):
    
    contour_index = compute_contour(ptr_array)    
    left_eye_index = connection_to_index(connection_dict["left eye"]) 
    right_eye_index = connection_to_index(connection_dict["right eye"])
    lip_index = connection_to_index(connection_dict["lips"])
    
    tri_index = left_eye_index + right_eye_index + lip_index + contour_index
    tri_points = ptr_array[tri_index]
    tri = Delaunay(tri_points)
    
    simplices = []
    LEI = list(range(len(left_eye_index)))
    REI = list(range( max(LEI)+1, max(LEI) + 1 + len(right_eye_index)))
    LPI = list(range( max(REI)+1, max(REI) + 1 + len(lip_index)))
    for a, b, c in tri.simplices:
        if a in LEI and b in LEI and c in LEI:
            continue
        if a in REI and b in REI and c in REI:
            continue
        if a in LPI and b in LPI and c in LPI:
            continue
        simplices.append([a, b, c])
        
    triangle_area = [ compute_triangle_area(*tri_points[triangle]) for triangle in simplices ]
    triangle_area = np.array(triangle_area)

    triangle_weight = triangle_area / triangle_area.sum()

    # find this gives a better balance
    weight = triangle_weight ** 2
    weight = weight / weight.sum()

    # weight = weight + triangle_weight
    # triangle_weight = weight / 2
        
    return tri_points, simplices, weight

def sample_from_triangle(pt1, pt2, pt3):
    """
    Random point on the triangle with vertices pt1, pt2 and pt3.
    """
    s, t = sorted([np.random.rand(), np.random.rand()])
    new_pt = [s * pt1[0] + (t-s)*pt2[0] + (1-t)*pt3[0],
                s * pt1[1] + (t-s)*pt2[1] + (1-t)*pt3[1]]
    new_pt = np.array(new_pt)
    return new_pt

def sample_from_simplices(tri_points, simplices, triangle_weight):
    
    tindex = np.random.choice(len(simplices), size=1, p=triangle_weight).item()
    triangle = tri_points[simplices[tindex]]
    new_ptr = sample_from_triangle(*triangle)
    new_ptr = new_ptr.astype(np.int)
    return new_ptr

def filter_points(ptr_array, img_shape, landmark_array, dist_thres):
    # remove outside image
    good = np.logical_and(ptr_array[:, 0] >= 0, ptr_array[:, 1] >=0)
    good = np.where(good)[0]
    ptr_array = ptr_array[good]
    
    good = np.logical_and(ptr_array[:, 0] < img_shape[1], ptr_array[:, 1]<img_shape[0])
    good = np.where(good)[0]
    ptr_array = ptr_array[good]
    
    # remove too close to landmark
    new = ptr_array[:, :, None]
    ctr = landmark_array.T[None, :, :]
    dist = np.abs(new - ctr)
    error = dist < (dist_thres) 
    too_close = np.logical_and(error[:, 0], error[:, 1])
    good = np.where(too_close.sum(1) <= 0)[0]
    ptr_array = ptr_array[good]
    
    return ptr_array

#----------------------------------------------------

def obtain_preprocess_triangles(img, landmark_list):

    lm_rel = landmark_to_array(landmark_list)
    lm_abs = [ normalized_to_pixel_coordinates(p[0], p[1], img) for p in lm_rel ]
    lm_abs = np.array(lm_abs)

    # generate triangle using outsize contour, eye, mouth 
    tri_points, simplices, triangle_weight = generate_triangle(lm_abs, CONNECTIONS)

    # landmark of contour, eye, eyebrow, mouth
    eyebrow_lm_index = connection_to_index(CONNECTIONS["left eyebrow"]) \
                        + connection_to_index(CONNECTIONS["right eyebrow"])
    named_lm = np.concatenate([tri_points, lm_abs[eyebrow_lm_index]], axis=0)

    return tri_points, simplices, triangle_weight, named_lm

def generate_filtered_patch(img, tri_params,
                        num_sample_point=10, size_ratio=10, filter_magic=1.5):

    tri_points, simplices, triangle_weight, named_lm = tri_params

    shape = tri_points.max(0) - tri_points.min(0) # h, w
    size = int ( min(shape) / size_ratio )
    while True:
        new_ptrs = []
        for i in range(num_sample_point):
            p = sample_from_simplices(tri_points, simplices, triangle_weight)
            new_ptrs.append(p)
        new_ptrs = np.array(new_ptrs)

        ptrs = filter_points(new_ptrs, img.shape, named_lm, size/filter_magic)

        if ptrs.shape[0] > 0:
            break

    ptr = ptrs[0]
    ptr_cs = [ min(ptr[0], img.shape[1]-size), min(ptr[1], img.shape[0]-size) ]
    ptr_cs = ( max(ptr_cs[0], size), max(ptr_cs[1], size) )

    patch = img[ptr_cs[1]-size:ptr_cs[1]+size, ptr_cs[0]-size:ptr_cs[0]+size]

    return ptr_cs, patch










