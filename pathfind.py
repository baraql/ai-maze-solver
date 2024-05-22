import cv2
import numpy as np
import pygame
import networkx as nx

def pygame_surface_to_opencv(pg_surface):
    rgb_array = pygame.surfarray.array3d(pg_surface)
    rgb_array = rgb_array.transpose([1, 0, 2])
    img_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    
    if pg_surface.get_masks()[3] != 0:
        alpha_array = pygame.surfarray.array_alpha(pg_surface)
        img_bgra = cv2.merge((img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2], alpha_array))
        return img_bgra

    return img_bgr

def overlay_images(base_img, overlay_img):
    if overlay_img.shape[2] == 3:
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2BGRA)
    
    if base_img.shape[2] == 3:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2BGRA)

    alpha_overlay = overlay_img[:, :, 3] / 255.0
    alpha_base = base_img[:, :, 3] / 255.0
    alpha_out = 1 - (1 - alpha_overlay) * (1 - alpha_base)
    
    for channel in range(3):
        base_img[:, :, channel] = (overlay_img[:, :, channel] * alpha_overlay + base_img[:, :, channel] * alpha_base * (1 - alpha_overlay)) / alpha_out
    
    base_img[:, :, 3] = alpha_out * 255

    return base_img

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    return binary

def skeletonize_with_cv2(binary_img):
    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    skeleton = cv2.ximgproc.thinning(closing, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return skeleton

def graph_from_skeleton(skel):
    graph = nx.Graph()
    for r in range(skel.shape[0]):
        for c in range(skel.shape[1]):
            if skel[r, c]:
                for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1] and skel[nr, nc]:
                        graph.add_edge((r, c), (nr, nc))
    return graph

def find_path(graph, start, end):
    return nx.dijkstra_path(graph, start, end)

def find_closest_point_in_graph(graph, point):
    min_distance = float('inf')
    closest_node = None
    for node in graph.nodes:
        dist = (node[0] - point[0]) ** 2 + (node[1] - point[1]) ** 2
        if dist < min_distance:
            min_distance = dist
            closest_node = node
    return closest_node

def opencv_to_pygame(cv_img):
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(img_rgb.transpose(1, 0, 2))

def process_and_draw_path(cv_img, start, end):
    cv2.imshow('Original Image', cv_img)
    cv2.waitKey(0)

    binary = preprocess_image(cv_img)
    cv2.imshow('Preprocessed Image', binary)
    cv2.waitKey(0)

    skeleton = skeletonize_with_cv2(binary)
    cv2.imshow('Skeleton Image', skeleton)
    cv2.waitKey(0)

    graph = graph_from_skeleton(skeleton)

    start = find_closest_point_in_graph(graph, start)
    end = find_closest_point_in_graph(graph, end)
    path = find_path(graph, start, end)

    for i in range(len(path) - 1):
        pt1 = path[i][::-1]
        pt2 = path[i + 1][::-1]
        cv2.line(cv_img, pt1, pt2, (0, 255, 0), 3)
        cv2.imshow('Path Drawing', cv_img)
        cv2.waitKey(0)

    return opencv_to_pygame(cv_img)

