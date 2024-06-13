from ultralytics import YOLO

class Model:
    def __init__(self, corner_detector_path, player_detector_path) -> None:
        self.corner_detector = YOLO(corner_detector_path)
        self.player_detector = YOLO(player_detector_path)

    def __call__(self, img_path, save=False):
            res_corner = self.corner_detector(img_path, save=save)[0]
            res_player = self.player_detector(img_path, save=save)[0]
            corner_bbox, corner_cls = res_corner.boxes.xywhn.cpu().detach().numpy(), res_corner.boxes.cls.cpu().detach().numpy()
            player_bbox, player_cls = res_player.boxes.xywhn.cpu().detach().numpy(), res_player.boxes.cls.cpu().detach().numpy()
            tmp = player_bbox[:,2:]
            player_bbox = player_bbox[:, :2]
            player_bbox[:,1] = player_bbox[:,1] + tmp[:,1] / 2
            corner_bbox = corner_bbox[:, :2]
            return corner_bbox, corner_cls, player_bbox, player_cls
