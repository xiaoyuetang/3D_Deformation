import pandas as pd
import numpy as np

from imutils import face_utils
import dlib
import cv2

import eos
import plotly.graph_objects as go


def pts2array(dict):
    pts = []
    for key, value in dict.items():
        pt = [value['x'],value['y']]
        pts.append(pt)
    return np.array(pts)


def generate_68pts(image_path):
    p = "data/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        pts = []
        landmarks = []
        # Draw on our image, all the finded cordinate points (x,y)
        for index, (x, y) in enumerate(shape):
            pt = [x,y]
            pts.append(pt)
            landmarks.append(eos.core.Landmark(str(index), [float(x), float(y)]))

        if i>0:
            break

    return np.array(pts), landmarks


def cal_pq(P68, mesh_v=None):
    '''
    第5步的p:
    Pa68 与mesh_a_v中对应的68个3D点坐标
    第5步的q:
    Pb68 与mesh_b_v中对应的68个3D点坐标
    '''
    zeros = np.array([0]*68).reshape(68,1)
    P68 = np.append(P68, zeros, axis=1)

    out = []
    for i, pts in enumerate(P68):
        smallest_dist = float('inf')
        smallest_j = float('inf')
        for j, vertex in enumerate(mesh_v):
            squared_dist = np.sum((pts-vertex)**2, axis=0)
            dist = np.sqrt(squared_dist)
            if dist < smallest_dist:
                smallest_dist = dist
                smallest_j = j
        out.append(mesh_v[smallest_j])
    return np.array(out)

def generate_mesh_v(image_path, landmarks):
    image = cv2.imread(image_path)
    image_width, image_height, _ = image.shape

    model = eos.morphablemodel.load_model("share/sfm_shape_3448.bin")
    blendshapes = eos.morphablemodel.load_blendshapes("share/expression_blendshapes_3448.bin")
    # Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
    morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                                        color_model=eos.morphablemodel.PcaModel(),
                                                                        vertex_definitions=None,
                                                                        texture_coordinates=model.get_texture_coordinates())
    landmark_mapper = eos.core.LandmarkMapper('share/ibug_to_sfm.txt')
    edge_topology = eos.morphablemodel.load_edge_topology('share/sfm_3448_edge_topology.json')
    contour_landmarks = eos.fitting.ContourLandmarks.load('share/ibug_to_sfm.txt')
    model_contour = eos.fitting.ModelContour.load('share/sfm_model_contours.json')

    (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(morphablemodel_with_expressions,
        landmarks, landmark_mapper, image_width, image_height, edge_topology, contour_landmarks, model_contour)

    mesh_v = mesh.vertices
    return np.array(mesh_v)


def mls_affine_deformation_3d(p, q, pts, alpha=1):
    '''
    Calculate the affine deformation of 3d points.
    '''
    ctrls = p.shape[0]
    np.seterr(divide='ignore')
    new_pts = []
    for i, v in enumerate(pts):
        wi = 1.0 / np.sum((p - v) ** 2, axis=1) ** alpha
        wi[wi == np.inf] = 2**31-1

        pstar = np.sum(p.T * wi, axis=1) / np.sum(wi)
        qstar = np.sum(q.T * wi, axis=1) / np.sum(wi)

        phat = p - pstar
        qhat = q - qstar

        reshaped_phat1 = phat.reshape(ctrls, 3, 1)
        reshaped_phat2 = phat.reshape(ctrls, 1, 3)

        reshaped_wi = wi.reshape(ctrls, 1, 1)
        pTwp = np.sum(reshaped_phat1 * reshaped_wi * reshaped_phat2, axis=0)

        try:
            inv_pTwp = np.linalg.inv(pTwp)
        except np.linalg.linalg.LinAlgError:
            if np.linalg.det(pTwp) < 1e-8:
                new_v = v + qstar - pstar
                return new_v
            else:
                raise
        mul_left = v - pstar
        mul_right = np.sum(reshaped_phat1 * reshaped_wi * qhat[:, np.newaxis, :], axis=0)
        new_v = np.dot(np.dot(mul_left, inv_pTwp), mul_right) + qstar
        new_pts.append(new_v)
    return np.array(new_pts)


def display_mesh(mesh_v):
    x, y, z = mesh_v.T

    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50)])
    fig.show()


if __name__ == "__main__":

    # 1.找一张正脸图像a,一张稍微有点侧的人脸图像b;
    # 2.用PGFaceEngine 检测a的2D点位Pa，b的2D点位Pb；
    Pa_df = pd.read_json("data/Claire_Danes_51.json")
    Pa = pts2array(Pa_df['faces'][0]['points'])

    Pb_df = pd.read_json("data/Claire_Danes_52.json")
    Pb = pts2array(Pb_df['faces'][0]['points'])

    # 3.按eos要求提取Pa的68点Pa68以及Pb68;
    Pa68, landmarks_a = generate_68pts("data/Claire_Danes_51.jpg")
    Pb68, landmarks_b = generate_68pts("data/Claire_Danes_52.jpg")

    # 4.用eos生成Pa,Pb的3D mesh: mesh_a，mesh_b;(其中的顶点用mesh_a_v, mesh_b_v表示)
    mesh_a_v = generate_mesh_v(image_path="data/Claire_Danes_51.jpg", landmarks=landmarks_a)
    mesh_b_v = generate_mesh_v(image_path="data/Claire_Danes_52.jpg", landmarks=landmarks_b)

    # 5.编写最小二乘变形函数并调用： mesh_b_v_new = morph(mesh_a_v，p,q)
    p = cal_pq(Pa68, mesh_a_v)
    q = cal_pq(Pb68, mesh_b_v)
    mesh_b_v_new = mls_affine_deformation_3d(p, q, pts=mesh_a_v)

    # 6.替换mesh_b中的mesh_b_v为mesh_b_v_new 并将新的mesh_b画在图b上看效果咋样
    display_mesh(mesh_b_v_new)
    display_mesh(mesh_b_v)
