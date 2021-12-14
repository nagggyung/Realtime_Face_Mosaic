import sys
import os

sys.path.append(os.path.join(sys.path[0], 'MTCNN'))
import argparse
import torch
from torchvision import transforms as trans
from utils.util import *
from utils.align_trans import *
from MTCNN import create_mtcnn_net
from face_model import MobileFaceNet, l2_norm
from facebank import load_facebank, prepare_facebank
import cv2
import time



def resize_image(img, scale):
    """
        resize image
    """
    height, width, channel = img.shape
    new_height = int(height * scale)  # resized new height
    new_width = int(width * scale)  # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
    return img_resized


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='face detection demo')
    parser.add_argument('-th', '--threshold', help='threshold score to decide identical faces', default=60, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true",
                        default=False)
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true", default=False)
    parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true", default=True)
    parser.add_argument("--scale", dest='scale', help="input frame scale to accurate the speed", default=0.5,
                        type=float)
    parser.add_argument('--mini_face', dest='mini_face', help=
    "Minimum face to be detected. derease to increase accuracy. Increase to increase speed",
                        default=20, type=int)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    detect_model = MobileFaceNet(512).to(device)  # embeding size is 512 (feature vector)
    detect_model.load_state_dict(torch.load('Weights/MobileFace_Net', map_location=lambda storage, loc: storage))
    print('MobileFaceNet face detection model generated')
    detect_model.eval()

    if args.update:
        targets, names = prepare_facebank(detect_model, path='facebank', tta=args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(path='facebank')
        #print(targets, names)
        print('facebank loaded')
        # targets: number of candidate x 512

    cap = cv2.VideoCapture(0)
    while True:
        isSuccess, frame = cap.read()
        if isSuccess:
            try:
                start_time = time.time()
                input = resize_image(frame, args.scale)

                bboxes, landmarks = create_mtcnn_net(input, 32, device,
                                                     p_model_path='MTCNN/weights/pnet_Weights',
                                                     r_model_path='MTCNN/weights/rnet_Weights',
                                                     o_model_path='MTCNN/weights/onet_Weights')

                #print("bbox1 : ", bboxes)
                #print()
                #print("landmarks: ", landmarks)

                if bboxes != []:
                    bboxes = bboxes / args.scale
                    landmarks = landmarks / args.scale

                #print("bbox2 : ", bboxes)
                #print("landmarks2: ", landmarks)

                faces = Face_alignment(frame, default_square=True, landmarks=landmarks)

                embs = []
                test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

                for img in faces:
                    if args.tta:
                        mirror = cv2.flip(img, 1)
                        emb = detect_model(test_transform(img).to(device).unsqueeze(0))
                        emb_mirror = detect_model(test_transform(mirror).to(device).unsqueeze(0))
                        embs.append(l2_norm(emb + emb_mirror))
                    else:
                        embs.append(detect_model(test_transform(img).to(device).unsqueeze(0)))

                source_embs = torch.cat(embs)  # number of detected faces x 512

                diff = source_embs.unsqueeze(-1) - targets.transpose(1, 0).unsqueeze(
                    0)  # i.e. 3 x 512 x 1 - 1 x 512 x 2 = 3 x 512 x 2
                dist = torch.sum(torch.pow(diff, 2), dim=1)  # number of detected faces x numer of target faces
                minimum, min_idx = torch.min(dist, dim=1)  # min and idx for each row
                min_idx[minimum > ((args.threshold - 156) / (-80))] = -1  # if no match, set idx to -1
                score = minimum
                results = min_idx

                # convert distance to score dis(0.7,1.2) to score(100,60)
                score_100 = torch.clamp(score * -80 + 156, 0, 100)
                FPS = 1.0 / (time.time() - start_time)
                cv2.putText(frame,'FPS: {:.1f}'.format(FPS) ,(25, 25), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)

                for i, b in enumerate(bboxes):
                   # print(i, b)
                    x = int(b[0])
                    y = int(b[1])
                    w = int(b[2])
                    h = int(b[3])
                    if results[i] != -1:
                        cv2.rectangle(frame, (x,y), (w, h), (0,155,255),2)
                        cv2.putText(frame, names[results[i] + 1] + ' score:{:.0f}'.format(score_100[i]), (x,y-25), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2, cv2.LINE_AA)
                        # blur
                        roi = frame[y:h, x:w]
                        roi = cv2.blur(roi, (20, 20))
                        frame[y:h, x:w] = roi
                    else:
                        cv2.rectangle(frame, (x, y), (w, h), (0, 155, 255), 2)
                        cv2.putText(frame, names[results[i] + 1], (x, y - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2, cv2.LINE_AA)

            except:
                print('detect error')

        cv2.imshow('video', frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
