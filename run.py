import cv2
import os
import numpy as np
import torch
import time
import argparse
from Sim3DR.renderer import render_fvr
import threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


stop_flag = False


def distance(x, y):
    return np.sqrt(((x - y) ** 2).sum())


def norm(x):
    return x / np.sqrt((x ** 2).sum())


class FrameLoader(threading.Thread):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mode = self._determine_mode()

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        # !!!!! Note: Maybe you want to use the video mode for a imagefolder !!!!!
        running_mode = VisionRunningMode.VIDEO if self.mode == 'video' or self.mode == 'webcam' else VisionRunningMode.IMAGE
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='data/face_landmarker.task'),
            running_mode=running_mode,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1)
        self.face_tracker = vision.FaceLandmarker.create_from_options(options)
        
        self.cap = None
        if self.mode == 'video':
            self.cap = cv2.VideoCapture(self.args.input)
        elif self.mode == 'webcam':
            self.cap = cv2.VideoCapture(0)
        self.current_index = 0
        self.frames_queue = []
        self.boxes_queue = []
        self.frame_names_queue = []
        self.done = False
        self.max_queue_size = 3

    def detect_lms(self, img):
        face_image = mp.Image(mp.ImageFormat.SRGB, img.astype(np.uint8))
        # !!!!! Note: Maybe you want to use the video mode for a imagefolder !!!!!
        if self.mode == 'video' or self.mode == 'webcam':
            results = self.face_tracker.detect_for_video(face_image, self.current_index * 33)
        else:
            results = self.face_tracker.detect(face_image)
        if not results.face_landmarks:
            return []
        else:
            lms = results.face_landmarks[0]
            lms = np.array([(lmk.x, lmk.y) for lmk in lms])
            lms[:, 0] = lms[:, 0] * img.shape[1]
            lms[:, 1] = lms[:, 1] * img.shape[0]
            vec_norm = norm(lms[362, :2] - lms[263, :2])
            leyex = np.dot(lms[473] - (lms[263, :2] + lms[362, :2]) / 2, vec_norm) / distance(lms[362, :2], lms[263, :2]) * 3
            leyey = np.dot(lms[473] - (lms[263, :2] + lms[362, :2]) / 2, vec_norm[[1, 0]]) / distance(lms[362, :2], lms[263, :2]) * -1.5
            vec_norm = norm(lms[33, :2] - lms[133, :2])
            reyex = np.dot(lms[468] - (lms[33, :2] + lms[133, :2]) / 2, vec_norm) / distance(lms[33, :2], lms[133, :2]) * 3
            reyey = np.dot(lms[468] - (lms[33, :2] + lms[133, :2]) / 2, vec_norm[[1, 0]]) / distance(lms[33, :2], lms[133, :2]) * -1.5
        return [[np.min(lms[:, 0]), np.min(lms[:, 1]), np.max(lms[:, 0]), np.max(lms[:, 1])], [leyey, leyex, reyey, reyex]]

    def _determine_mode(self):
        if self.args.input.endswith('.mp4') or self.args.input.endswith('.MP4') or self.args.input.endswith('.avi'):
            return 'video'
        elif self.args.input.endswith('.png') or self.args.input.endswith('.jpg') or self.args.input.endswith('.jpeg'):
            return 'image'
        elif self.args.input.endswith('webcam'):
            return 'webcam'
        return 'imagefolder'

    def run(self):
        if self.mode == 'video' or self.mode == 'webcam':
            while True:
                if stop_flag:
                    break
                if len(self.frames_queue) < self.max_queue_size:
                    ret, frame = self.cap.read()
                    if ret:
                        frame_name = f"frame_{str(self.current_index).zfill(5)}.jpg"
                        self.current_index += 1
                        self.boxes_queue.append(self.detect_lms(frame[:, :, ::-1]))
                        self.frames_queue.append(frame[:, :, ::-1])
                        self.frame_names_queue.append(frame_name)
                    else:
                        self.cap.release()
                        self.done = True
                        break
                else:
                    time.sleep(0.01)
                    continue
        elif self.mode == 'image':
            frame = cv2.imread(self.args.input)[:, :, :3]
            file_name = os.path.basename(self.args.input)
            if frame is not None:
                self.boxes_queue.append(self.detect_lms(frame[:, :, ::-1]))
                self.frames_queue.append(frame[:, :, ::-1])
                self.frame_names_queue.append(file_name)
            self.done = True
        elif self.mode == 'imagefolder':
            root, dirs, files = next(os.walk(self.args.input))
            for file in files:
                if stop_flag:
                    break
                if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                    file_path = os.path.join(root, file)
                    frame = cv2.imread(file_path)[:, :, :3]
                    if frame is not None:
                        if len(self.frames_queue) < self.max_queue_size:
                            self.boxes_queue.append(self.detect_lms(frame[:, :, ::-1]))
                            self.frames_queue.append(frame[:, :, ::-1])
                            self.frame_names_queue.append(file)
                        else:
                            time.sleep(0.01)
                            continue
            self.done = True

    def get_next_frame(self):
        if len(self.frames_queue) > 0:
            frame = self.frames_queue.pop(0)
            frame_name = self.frame_names_queue.pop(0)
            boxes = self.boxes_queue.pop(0)
            return frame, frame_name, boxes
        return None, None, None


def load_faceverse(device):
    from faceversev4 import FaceVerseRecon
    fvr = FaceVerseRecon(
        "data/faceverse_v4_2.npy",
        "data/faceverse_resnet50.pth",
        device
    )
    return fvr


def ply_from_array(points, faces, output_file):

    num_points = len(points)
    num_triangles = len(faces)

    header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
element face {}
property list uchar int vertex_indices
end_header\n'''.format(num_points, num_triangles)

    with open(output_file,'w') as f:
        f.writelines(header)
        for item in points:
            f.write("{0:0.6f} {1:0.6f} {2:0.6f}\n".format(item[0], item[1], item[2]))

        for item in faces:
            number = len(item)
            row = "{0}".format(number)
            for elem in item:
                row += " {0} ".format(elem)
            row += "\n"
            f.write(row)


def ply_from_array_color(points, colors, faces, output_file):

    num_points = len(points)
    num_triangles = len(faces)

    header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face {}
property list uchar int vertex_indices
end_header\n'''.format(num_points, num_triangles)

    with open(output_file,'w') as f:
        f.writelines(header)
        index = 0
        for item in points:
            f.write("{0:0.6f} {1:0.6f} {2:0.6f} {3} {4} {5}\n".format(item[0], item[1], item[2],
                                                        colors[index, 0], colors[index, 1], colors[index, 2]))
            index = index + 1

        for item in faces:
            number = len(item)
            row = "{0}".format(number)
            for elem in item:
                row += " {0} ".format(elem)
            row += "\n"
            f.write(row)


def process_one_batch(args, frameloader, fvr, box_batch, frame_batch, name_batch, cache_frame, cache_param, end=False):
    if len(name_batch) > 0:
        box_batch = np.stack(box_batch)
        eye_batch = box_batch[:, 1]
        box_batch = box_batch[:, 0:1]
        frame_batch = np.stack(frame_batch)

        coeffs, bbox_list = fvr.process_imgs(frame_batch, box_batch)
        # force to get pupil pos from mediapipe
        coeffs[:, -4:] = torch.from_numpy(eye_batch).to(coeffs.device)
        # a second forward for smooth bbox
        # _, vs_proj, _, _ = fvr.from_coeffs(coeffs, bbox_list)
        # lms = vs_proj[:, fvr.fvd['keypoints']]
        # coeffs, _ = fvr.process_imgs(frame_batch, lms)

    if args.visual:
        for index, name in enumerate(name_batch):
            if args.smooth:
                if len(cache_param) == 1:
                    coeffs_this = cache_param[-1]['coeffs']
                    bbox_list_this = cache_param[-1]['bbox_list']
                    framet_this = cache_frame[-1]['frame']
                    namet_this = cache_frame[-1]['name']
                elif len(cache_param) == 2:
                    coeffs_this = (cache_param[-2]['coeffs'] + cache_param[-1]['coeffs'] + coeffs[index:index+1]) / 3
                    bbox_list_this = cache_param[-1]['bbox_list']
                    framet_this = cache_frame[-1]['frame']
                    namet_this = cache_frame[-1]['name']
                else:
                    coeffs_this = None
            else:
                coeffs_this = coeffs[index:index+1]
                bbox_list_this = bbox_list[index:index+1]
                framet_this = frame_batch[index]
                namet_this = name

            if coeffs_this is not None:
                vs, vs_proj, normal, colors = fvr.from_coeffs(coeffs_this, bbox_list_this)
                rgb, depth = render_fvr(framet_this, vs_proj[0], fvr.fvd["tri"], normal[0], colors[0])

                save_img = np.concatenate([framet_this[:, :, ::-1], rgb[:, :, ::-1]], axis=0)
                bbox_list_this = bbox_list_this.astype(np.int32)
                cv2.rectangle(save_img, (bbox_list_this[0, 0], bbox_list_this[0, 1]), (bbox_list_this[0, 2], bbox_list_this[0, 3]), (0, 255, 0), 2)
                if args.save_results:
                    cv2.imwrite(os.path.join(args.output, namet_this), save_img)
                    headparam = {'coeffs': coeffs_this.cpu().numpy(), 'bbox_list': bbox_list_this}
                    # for loading the param: f=np.load('xxx.npy', allow_pickle=True).item(), you can get a dict f
                    np.save(os.path.join(args.output, namet_this.split('.')[0]), headparam)
                    if args.save_ply:
                        ply_from_array_color(vs[0], (colors[0] * 255).astype(np.uint8), fvr.fvd["tri"], os.path.join(args.output, namet_this.split('.')[0] + '.ply'))

                if frameloader.mode == 'webcam':
                    cv2.imshow('Webcam', save_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        frameloader.done = True
    
            if args.smooth:
                headparam = {'coeffs': coeffs[index:index+1], 'bbox_list': bbox_list[index:index+1]}
                cache_param.append(headparam)
                frame_t = {'frame': frame_batch[index], 'name': name}
                cache_frame.append(frame_t)
                if len(cache_param) > 2:
                    cache_param.pop(0)
                    cache_frame.pop(0)
        if end:
            while len(cache_param) > 1:
                coeffs_this = cache_param[-1]['coeffs']
                bbox_list_this = cache_param[-1]['bbox_list']
                framet_this = cache_frame[-1]['frame']
                namet_this = cache_frame[-1]['name']

                vs, vs_proj, normal, colors = fvr.from_coeffs(coeffs_this, bbox_list_this)
                rgb, depth = render_fvr(framet_this, vs_proj[0], fvr.fvd["tri"], normal[0], colors[0])

                save_img = np.concatenate([framet_this[:, :, ::-1], rgb[:, :, ::-1]], axis=0)
                bbox_list_this = bbox_list_this.astype(np.int32)
                cv2.rectangle(save_img, (bbox_list_this[0, 0], bbox_list_this[0, 1]), (bbox_list_this[0, 2], bbox_list_this[0, 3]), (0, 255, 0), 2)
                if args.save_results:
                    cv2.imwrite(os.path.join(args.output, namet_this), save_img)
                    headparam = {'coeffs': coeffs_this.cpu().numpy(), 'bbox_list': bbox_list_this}
                    # for loading the param: f=np.load('xxx.npy', allow_pickle=True).item(), you can get a dict f
                    np.save(os.path.join(args.output, namet_this.split('.')[0]), headparam)
                    if args.save_ply:
                        ply_from_array_color(vs[0], (colors[0] * 255).astype(np.uint8), fvr.fvd["tri"], os.path.join(args.output, namet_this.split('.')[0] + '.ply'))

                if frameloader.mode == 'webcam':
                    cv2.imshow('Webcam', save_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        # !!!! Not used! Please use ctrl+c to stop the webcamera !!!!
                        frameloader.done = True
                        exit()
                cache_param.pop(0)
                cache_frame.pop(0)
            while len(cache_param) > 0:
                cache_param.pop(0)
                cache_frame.pop(0)


def run(args, fvr):
    frameloader = FrameLoader(args)
    frameloader.start()

    os.makedirs(args.output, exist_ok=True)

    if frameloader.mode == 'webcam':
        args.batch = 1

    # for smooth
    box_buffer = []
    frame_buffer = []
    name_buffer = []

    # for network input
    box_batch = []
    frame_batch = []
    name_batch = []

    frame_num = 0
    start_time_total = time.time()

    cache_frame = []
    cache_param = []

    while True:
        if stop_flag:
            break
        frame, frame_name, boxes = frameloader.get_next_frame()
        if frame is None:
            if frameloader.done:
                break
            time.sleep(0.01)
            continue
        
        if len(boxes) > 0:
            box = np.array(boxes).astype(np.float32)[0, :4]
            eyes = np.array(boxes).astype(np.float32)[1, :4]

            width = box[2] - box[0]
            height = box[3] - box[1]
            side_length = max(width, height)
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
            box[0] = center_x - side_length // 2
            box[1] = center_y - side_length // 2
            box[2] = center_x + side_length // 2
            box[3] = center_y + side_length // 2

            if args.smooth:
                box_buffer.append(np.stack([box, eyes]))
                frame_buffer.append(frame)
                name_buffer.append(frame_name)
                if len(box_buffer) > 3:
                    box_buffer.pop(0)
                    frame_buffer.pop(0)
                    name_buffer.pop(0)
                smoothed_box = np.mean(box_buffer, axis=0) if len(box_buffer) == 3 else box_buffer[0]
                frame_this = frame_buffer[1] if len(box_buffer) > 1 else frame_buffer[0]
                name_this = name_buffer[1] if len(box_buffer) > 1 else name_buffer[0]
            else:
                smoothed_box = np.stack([box, eyes])
                frame_this = frame
                name_this = frame_name

            if len(box_batch) < args.batch:
                box_batch.append(smoothed_box)
                frame_batch.append(frame_this)
                name_batch.append(name_this)
            else:
                box_batch.append(smoothed_box)
                frame_batch.append(frame_this)
                name_batch.append(name_this)
                process_one_batch(args, frameloader, fvr, box_batch, frame_batch, name_batch, cache_frame, cache_param)
                box_batch = []
                frame_batch = []
                name_batch = []
        else:
            if args.smooth and len(box_buffer) > 0:
                smoothed_box = box_buffer[-1]
                frame_this = frame_buffer[-1]
                name_this = name_buffer[-1]
                box_batch.append(smoothed_box)
                frame_batch.append(frame_this)
                name_batch.append(name_this)
                process_one_batch(args, frameloader, fvr, box_batch, frame_batch, name_batch, cache_frame, cache_param, end=True)
                box_batch = []
                frame_batch = []
                name_batch = []
            box_buffer = []
            frame_buffer = []
            name_buffer = []
            print("No face detected in", frame_name)
            if frameloader.mode == 'webcam':
                save_img = np.concatenate([frame[:, :, ::-1], np.zeros_like(frame)], axis=0)
                cv2.imshow('Webcam', save_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # !!!! Not used! Please use ctrl+c to stop the webcamera !!!!
                    frameloader.done = True
            continue

        frame_num += 1
        frame_time = (time.time() - start_time_total) / frame_num
        if frame_num % 100 == 0:
            print(f"Processing time for frame {frame_name}: {frame_time:.4f} seconds", frame.shape)

    # process the last one frame
    if args.smooth and len(box_buffer) > 0:
        smoothed_box = box_buffer[-1]
        frame_this = frame_buffer[-1]
        name_this = name_buffer[-1]
        box_batch.append(smoothed_box)
        frame_batch.append(frame_this)
        name_batch.append(name_this)

    process_one_batch(args, frameloader, fvr, box_batch, frame_batch, name_batch, cache_frame, cache_param, end=True)

    frameloader.join()
    end_time_total = time.time()
    total_time = end_time_total - start_time_total
    print(f"Total running time: {total_time:.4f} seconds")
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="example/input/test.mp4")  # video or image or imagefolder or 'webcam'
    parser.add_argument("--output", type=str, default="example/output")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--smooth", type=bool, default=True)
    parser.add_argument("--save_results", type=bool, default=False)
    parser.add_argument("--save_ply", type=bool, default=False)
    parser.add_argument("--visual", type=bool, default=True)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    fvr = load_faceverse(device)

    try:
        run(args, fvr)
    except KeyboardInterrupt:
        stop_flag = True
        print("Running script has been killed!")
