import random
from itertools import combinations, product
import numpy as np
import cv2
import time

class Point:
    def __init__(self, x, y, type):
        self.x, self.y = x, y
        self.type = type

def list2points(bbox, cls):
    points = []
    for i in range(len(bbox)):
        points.append(Point(bbox[i, 0], bbox[i, 1], cls[i]))
    return points

def points2list(points):
    bbox = []
    cls = []
    for p in points:
        bbox.append([p.x, p.y])
        cls.append(p.type)
    return np.array(bbox), np.array(cls)
        


class Solver:
    def __init__(self) -> None:
        self.points = {
			4: [Point(0, 0, 4)], # 1
			3: [Point(-41.5, 0, 3), Point(41.5, 0, 3)], # 2
			2: [Point(-52.5, -3.66, 2), Point(-52.5, 3.66, 2), Point(52.5, -3.66, 2), Point(52.5, 3.66, 2)], # 4
			12:  [Point(-32.35, 0, 12), Point(32.35, 0, 12)], # 2
			11:  [Point(-52.5, -9.15, 11), Point(-52.5, 9.15, 11), Point(52.5, -9.15, 11), Point(52.5, 9.15, 11)], # 4
			10:  [Point(-9.15, 0, 10), Point(9.15, 0, 10)], # 2
			9:  [Point(-52.5, -20.15, 9), Point(-52.5, 20.15, 9), Point(52.5, -20.15, 9), Point(52.5, 20.15, 9)], # 4
			8:  [Point(0, -34, 8), Point(0, 34, 8)], # 2
			7:  [Point(-52.5, -34, 7), Point(-52.5, 34, 7), Point(52.5, -34, 7), Point(52.5, 34, 7)], # 4
			6:  [Point(-47, -9.15, 6), Point(-47, 9.15, 6), Point(47, -9.15, 6), Point(47, 9.15, 6)], # 4
			5:  [Point(-36, -7.32, 5), Point(-36, 7.32, 5), Point(36, -7.32, 5), Point(36, 7.32, 5)], # 4
			1:  [Point(0, -9.15, 1), Point(0, 9.15, 1)], # 2
			0:  [Point(-36, -20.15, 0), Point(-36, 20.15, 0), Point(36, -20.15, 0), Point(36, 20.15, 0)] # 4
		}

    def xycls(self, x, y, cls):
        scores = []
        for p in self.points[cls]:
            scores.append((p.x - x)**2 + (p.y - y)**2)
        return min(scores)

    def __are_collinear(self, p1:Point, p2:Point, p3:Point, tolerance=1e-4):
        v1 = (p2.x - p1.x, p2.y - p1.y)
        v2 = (p3.x - p1.x, p3.y - p1.y)
        corss_product = v1[0] * v2[1] - v1[1] * v2[0]
        return abs(corss_product) < tolerance

    def __choose_points(self, points, max_attempts=10000, num_selected_points=3):
        num_points = len(points)
        if num_points < 4:
            return None
        
        selected_points_arr = []
        attempt = 0
        N = 0

        while attempt < max_attempts and N < num_selected_points:
            selected_points = random.sample(points, 4)
            collinear = False
            for triplet in combinations(selected_points, 3):
                if self.__are_collinear(*triplet):
                    collinear = True
                    break
            if not collinear:
                selected_points_arr.append(selected_points)
                N += 1
            attempt += 1
        if len(selected_points_arr) == 0:
            return None
        return selected_points_arr
    
    def __compute_homography(self, src, dst):
        return cv2.findHomography(src, dst, cv2.RANSAC)[0]
	
    def __score_homography(self, hom, src, cls):
        prj = cv2.perspectiveTransform(src[:, None,:], hom)[:,0,:]
        sc = 0
        for i in range(prj.shape[0]):
            x, y = prj[i]
            sc += self.xycls(x, y, cls[i])
        return -sc
    
    def __perm2point(self, perm):
        N = []
        for _ in perm :
            N.extend(list(_))
        return np.array(N)

    def __solve(self, points, whole_points):
        src, cls = points2list(points)
        whole_src, whole_cls = points2list(whole_points)
        permuts = []
        asort = np.argsort(cls)
        src = src[asort, :]
        cls = cls[asort]
        unique_cls, freq_cls = np.unique(cls, return_counts=True)
        for i, j in zip(unique_cls, freq_cls):
            permuts.append(list(combinations(self.points[int(i)], j)))
        permuts = list(product(*permuts))
        best_score = -float('inf')
        best_hom = None
        best_map = None
        for p in permuts:            
            perm = self.__perm2point(p)
            perm, _ = points2list(perm)            
            H = self.__compute_homography(src, perm)
            if H is not None:
                score = self.__score_homography(H, whole_src, whole_cls)
                if score >= best_score:
                    best_hom = H
                    best_score = score
                    best_map = perm
        return best_hom, best_map, best_score
    
    def solve(self, bbox, cls, num_selected_points=5):
        tic = time.time()
        points = list2points(bbox, cls)
        selected_points_arr = self.__choose_points(points, num_selected_points=num_selected_points)
        
        homos = []
        scores = []
        if selected_points_arr:
            for selected_points in selected_points_arr:
                hom_matrix, _, s = self.__solve(selected_points, points)
                scores.append(s)
                homos.append(hom_matrix)                
            print("Time (s):",time.time() - tic,"\nScore   :", max(scores))
            return homos[scores.index(max(scores))]
        else:
            return None

if __name__ == "__main__":
    src = np.array([[    0.34273,     0.46745],
       [    0.53001,     0.39316],
       [    0.60883,     0.50871],
       [    0.64262,      0.4418],
       [    0.52805,     0.68775],
       [   0.089158,     0.49562],
       [     0.3033,     0.55245],
       [    0.45601,     0.36371],
       [    0.73115,     0.48067],
       [    0.19511,      0.2554],
       [    0.32907,     0.31127],
       [    0.35264,     0.38518],
       [    0.95598,      0.5766]], dtype=np.float32)
    cls = np.array([          3,           2,           6,           2,           0,          12,           5,          11,          11,           7,           9,           6,           9], dtype=np.float32)
    solver = Solver()
    solver.solve(src, cls)
