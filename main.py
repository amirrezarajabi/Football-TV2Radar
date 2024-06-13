import numpy as np
from utils import Solver, Pitch, Point, Model
import cv2
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--corner", type=str)
parser.add_argument("--player", type=str)
parser.add_argument("--img", type=str)
args = parser.parse_args()

model = Model(corner_detector_path=args.corner, player_detector_path=args.player)


corner_bbox, corner_cls, player_bbox, player_cls = model(args.img, save=True)

solver = Solver()
hom = solver.solve(corner_bbox, corner_cls)


corners = cv2.perspectiveTransform(corner_bbox[:, None,:], hom)[:, 0, :]
cs = []
for i in range(corners.shape[0]):
    cs.append(Point(corners[i, 0], corners[i, 1]))

players = cv2.perspectiveTransform(player_bbox[:, None,:], hom)[:, 0, :]

ps = []
for i in range(players.shape[0]):
    ps.append(Point(players[i, 0], players[i, 1], c="red"))

p = Pitch(color="green")

p.show(corners=cs)
p.show(players=ps)
