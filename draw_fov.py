import math
import pygame
import numpy as np

FOV = np.float64(120.20219632277087)
COLOR = (255, 255, 255)

def draw_fov(surface, position, angle, radius):
    angle_rad = math.radians(angle)
    angle_rad -= math.pi / 2

    start_angle = angle_rad - math.radians(FOV / 2)
    end_angle = angle_rad + math.radians(FOV / 2)

    point1 = (position[0] + radius * math.cos(start_angle), position[1] + radius * math.sin(start_angle))
    point2 = (position[0] + radius * math.cos(end_angle), position[1] + radius * math.sin(end_angle))

    points = [position, point1, point2]

    pygame.draw.polygon(surface, COLOR, points)