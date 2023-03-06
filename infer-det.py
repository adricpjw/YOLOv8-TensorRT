from models import TRTModule  # isort:skip
import argparse
from pathlib import Path

import cv2
import torch
import time
from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list

    # for image in images:
    #     save_image = save_path / image.name
    #     bgr = cv2.imread(str(image))
    #     draw = bgr.copy()
    #     bgr, ratio, dwdh = letterbox(bgr, (W, H))
    #     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    #     tensor = blob(rgb, return_seg=False)
    #     dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
    #     tensor = torch.asarray(tensor, device=device)
    #     # inference
    #     data = Engine(tensor)

    #     bboxes, scores, labels = det_postprocess(data)
    #     bboxes -= dwdh
    #     bboxes /= ratio

    #     for (bbox, score, label) in zip(bboxes, scores, labels):
    #         bbox = bbox.round().int().tolist()
    #         cls_id = int(label)
    #         cls = CLASSES[cls_id]
    #         color = COLORS[cls]
    #         cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
    #         cv2.putText(draw,
    #                     f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
    #                     cv2.FONT_HERSHEY_SIMPLEX,
    #                     0.75, [225, 255, 255],
    #                     thickness=2)
    #     if args.show:
    #         cv2.imshow('result', draw)
    #         cv2.waitKey(0)
    #     else:
    #         cv2.imwrite(str(save_image), draw)


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    source = ''
    if (args.source == '0') :
        source = 0
    else:
        source = args.source

    video = cv2.VideoCapture(source)
    save_path = Path(args.out_dir)

    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)\

    while True:
        
        ret, img = video.read()

        if not ret:
            break
        
        start = time.time()
        height, width, channels = img.shape
        scale = 640 / max(height, width)
        img = cv2.resize(img, (round(scale * width), round(scale * height)))

        draw = img.copy()
        img, ratio, dwdh = letterbox(img, (W, H))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)
        # inference
        data = Engine(tensor)
        #print('--------------------')
        #print(len(data))
        #print(data)
        bboxes, scores, labels = det_postprocess(data)
        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label) in zip(bboxes, scores, labels):
           bbox = bbox.round().int().tolist()
           cls_id = int(label)
           cls = CLASSES[cls_id]
           print(bbox, score, label)
           color = COLORS[cls]
           cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
           cv2.putText(draw,
                       f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.75, [225, 255, 255],
                       thickness=2)
        print((1/(time.time()-start)), " fps")
        
        if args.show:
            cv2.imshow('result',draw)
            cv2.waitKey(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--source', type=str, help='Video or webcam source')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--out-dir',
                        type=str,
                        default='./output',
                        help='Path to output file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
