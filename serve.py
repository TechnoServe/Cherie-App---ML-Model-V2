import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.cluster.hierarchy import linkage, fcluster
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
class ClassifierModel(nn.Module):
    def __init__(self):
        super(ClassifierModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (B, 3, 128, 128)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 64x64
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 32x32
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # 16x16
        x = self.dropout(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)  # 8x8
        
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x  # shape: (B, 3)
    
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def adjust_gamma(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def non_max_suppression_fast(boxes, scores, overlapThresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return pick
 
def filter_detections_by_size(detections, std_threshold=3):
    # Calculate areas of all detections
    areas = [abs(d['box'][2] - d['box'][0]) * abs(d['box'][3] - d['box'][1]) for d in detections]
    print("non-filtered_detections:", len(detections))
    # Calculate mean and standard deviation of areas
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    max_area = np.max(areas)
    min_area = np.min(areas)

    print("Mean area:", mean_area)
    print("Std area:", std_area)
    print("Max area:", max_area)
    print("Min area:", min_area)
    
    # Define acceptable range
    min_area = 200
    max_area = (mean_area*std_threshold)
    max_area = max(max_area, 20000)

    print("Min area:", min_area)
    print("Max area:", max_area)
    
    # Filter detections
    filtered_detections = [
        d for d, area in zip(detections, areas) 
        if min_area <= area <= max_area
    ]

    print("filtered_detections:", len(filtered_detections))
    
    return filtered_detections


def efficient_multi_scale_detection(image_path, model, conf_threshold=0.1, iou_threshold=0.3):
    original_image = cv2.imread(image_path)
    h, w = original_image.shape[:2]

    scale_factors = [1.0, 1.5]
    patch_sizes = [(h//1, w//1), (h//2, w//2)]

    all_detections = []
    batch_inserts= []
    for scale in scale_factors:
        scaled_image = cv2.resize(original_image, (int(w*scale), int(h*scale)))
        
        enhanced_images = [
            scaled_image,
            apply_clahe(scaled_image),
            adjust_gamma(scaled_image, 0.8),
            adjust_gamma(scaled_image, 1.2)
        ]

        for img in enhanced_images:
            batch_inserts.append((img, scale, (0,0)))

            for patch_h, patch_w in patch_sizes:
                for i in range(0, img.shape[0] - patch_h + 1, patch_h // 2):
                    for j in range(0, img.shape[1] - patch_w + 1, patch_w // 2):
                        patch = img[i:i+patch_h, j:j+patch_w]
                        batch_inserts.append((patch, scale, (i, j)))

    batch_batch = [v[0] for v in batch_inserts]
    # split batch into smaller batches
    batch_size = 16
    mini_batches = [batch_batch[i:i + batch_size] for i in range(0, len(batch_batch), batch_size)]
    results = [model(bt, conf=conf_threshold) for bt in mini_batches]
    # flatten results
    results = [r for res in results for r in res]
    for details, r in zip(batch_inserts,results):
        img, scale, (i, j) = details
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        for box, score in zip(boxes, scores):
            adjusted_box = [
                (box[0] + j) / scale,
                (box[1] + i) / scale,
                (box[2] + j) / scale,
                (box[3] + i) / scale
            ]
            all_detections.append({
                'box': adjusted_box,
                'score': score,
                'scale': scale
            })
            
    print("All detections:", len(all_detections))
    all_detections = filter_detections_by_size(all_detections)
    boxes = np.array([d['box'] for d in all_detections])
    scores = np.array([d['score'] for d in all_detections])

    
    keep = non_max_suppression_fast(boxes, scores, iou_threshold)
    
    filtered_detections = [all_detections[i] for i in keep]

    if len(filtered_detections) == 0:
        print("Nothing detected")
        return [], original_image

    detection_features = np.array([[d['box'][0], d['box'][1], d['box'][2], d['box'][3], d['score'], d['scale']] for d in filtered_detections])
    Z = linkage(detection_features, 'ward')

    clusters = fcluster(Z, t=0.5, criterion='distance')

    final_results = []
    for cluster_id in np.unique(clusters):
        cluster_detections = [filtered_detections[i] for i in range(len(filtered_detections)) if clusters[i] == cluster_id]
        best_detection = max(cluster_detections, key=lambda x: x['score'])
        final_results.append(best_detection)

    final_results.sort(key=lambda x: x['score'], reverse=True)
    final_results = final_results.copy()

    # delete all rest to free memory and vram
    del all_detections
    del batch_inserts
    del results
    del filtered_detections
    del detection_features
    del Z
    del clusters
    del keep
    del boxes
    del scores

    torch.cuda.empty_cache()

    return final_results, original_image


def merge_close_boundaries(detections, iou_threshold=0.7):
    merged = []
    detections.sort(key=lambda x: x['score'], reverse=True)
    
    for det in detections:
        if not merged:
            merged.append(det)
        else:
            should_merge = False
            for i, m_det in enumerate(merged):
                iou = calculate_iou(det['box'], m_det['box'])
                if iou > iou_threshold:
                    merged[i]['box'] = [
                        min(det['box'][0], m_det['box'][0]),
                        min(det['box'][1], m_det['box'][1]),
                        max(det['box'][2], m_det['box'][2]),
                        max(det['box'][3], m_det['box'][3])
                    ]
                    merged[i]['score'] = max(det['score'], m_det['score'])
                    should_merge = True
                    break
            if not should_merge:
                merged.append(det)
    
    return merged

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection / float(area1 + area2 - intersection)
    return iou



model = YOLO("runs/detect/train8/weights/best.pt")
classifier = ClassifierModel()
classifier.load_state_dict(torch.load('classifier.pth', map_location='cpu'))
classifier = classifier.half().cuda()
classifier = classifier.eval()

def pre_process_image(image, noise_up=False):
    if noise_up:
        # create noise
        if random.random() < 0.5:
            noise = np.random.normal(0, 255, image.shape) * 0.05
            image = image + noise
            image = np.clip(image, 0, 255)
            image = image.astype(np.uint8)
        else:

            # add brightness
            image = image + ((((random.random()*2)-1) * 0.05)*255)
            image = np.clip(image, 0, 255)
            # convert to int
            image = image.astype(np.uint8)



    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    return image
import io
def markup_image(image_path):
    detections, original_image = efficient_multi_scale_detection(image_path, model)
    detections = merge_close_boundaries(detections)
    detections = filter_detections_by_size(detections)
    bean_images = []
    for det in detections:
        box = det['box']
        score = det['score']
        bean_images.append((original_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])], box, score))
    

    bean_images = [(pre_process_image(img, False), box, score)  for img, box, score in bean_images]

    batched_images = torch.stack([torch.Tensor(img) for img, _, _ in bean_images]).half().cuda()
    
    with torch.no_grad():
        outputs = classifier(batched_images)
        _, predicted_clsx = torch.max(outputs.data, 1)
    del outputs 
    del batched_images
    torch.cuda.empty_cache()
    
    # Visualize results
    result_image = original_image.copy()
    total_detection = len(bean_images)
    classes_count = [0, 0, 0]
    for i, (_, box, score) in enumerate(bean_images):
        colrs = [
            (50, 50, 255),
            (255, 50, 50),
            (50, 200, 50)
        ]
        # curr_score = (variance[i] - variance.min()) / (variance.max() - variance.min())
        sel_color = colrs[predicted_clsx[i].item()]
        classes_count[predicted_clsx[i].item()] += 1
        if not (score >= 0.05):
            sel_color = (0, 0, 0)
        cv2.rectangle(result_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),sel_color , 2)
    
    # percentage of each class
    classes_count = [(count/total_detection)*100 for count in classes_count]
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    # convert to jpg, with 20% quality
    result_image = Image.fromarray(result_image)
    # save it to buffer and load it with PIL
    buffr = io.BytesIO()
    result_image.save(buffr, format="JPEG", quality=20)
    result_image = Image.open(buffr)
    
    return result_image, classes_count



# Gradio interface setup
with gr.Blocks(title="TechnoServe test") as demo:
    gr.Markdown("## TechnoServe")
    gr.Markdown("This demo is for TechnoServe")

    file_input = gr.Image(label="Upload bean image", type="filepath", height=200)
    # image show
    output_gallery = gr.Image(label="Annotation View", height=400) 
    score_output = gr.Textbox(label="Scores Results")

    def process_image(image_path):
        # time start
        started = time.time()
        annotated_image, counts = markup_image(image_path)
        # time end
        ended = time.time()
        scores = "Overripe [dark red]: {:.2f}%\nRipe [blue]: {:.2f}%\nUnderripe [green]: {:.2f}%".format(*counts)

        # took x seconds
        print (f"Processing took {ended-started} seconds")
        return scores, annotated_image

    file_input.change(process_image, inputs=file_input, outputs=[score_output, output_gallery])

# Launch the Gradio interface
demo.launch(share=True)
