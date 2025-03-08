import os
from functools import partial
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import random
import datetime
import numpy as np
import re
import argparse

Image.MAX_IMAGE_PIXELS = None


class CoordinatesQuantizer(object):
    """
    Quantize coornidates (Nx2)
    """

    def __init__(self, mode, bins):
        self.mode = mode
        self.bins = bins

    def quantize(self, coordinates: torch.Tensor, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size  # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        assert coordinates.shape[-1] == 2, "coordinates should be shape (N, 2)"
        x, y = coordinates.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == "floor":
            quantized_x = (x / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_y = (y / size_per_bin_h).floor().clamp(0, bins_h - 1)

        elif self.mode == "round":
            raise NotImplementedError()

        else:
            raise ValueError("Incorrect quantization type.")

        quantized_coordinates = torch.cat((quantized_x, quantized_y), dim=-1).int()

        return quantized_coordinates

    def dequantize(self, coordinates: torch.Tensor, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size  # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        assert coordinates.shape[-1] == 2, "coordinates should be shape (N, 2)"
        x, y = coordinates.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == "floor":
            # Add 0.5 to use the center position of the bin as the coordinate.
            dequantized_x = (x + 0.5) * size_per_bin_w
            dequantized_y = (y + 0.5) * size_per_bin_h

        elif self.mode == "round":
            raise NotImplementedError()

        else:
            raise ValueError("Incorrect quantization type.")

        dequantized_coordinates = torch.cat((dequantized_x, dequantized_y), dim=-1)

        return dequantized_coordinates


class RSDataset(Dataset):

    def __init__(self, args, data):
        self.data = data
        self.root_path = args.dataset_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ann = self.data[idx]

        image_dirs = ann["images"]
        image_id = image_dirs[0]
        crop = ann["crop"]
        task_TAG = ann["conversations"][0]["content"].split("\n")[1]
        prompt = ann["conversations"][0]["content"].replace(
            "<image>\n{}\n".format(task_TAG), ""
        )
        label = ann["conversations"][1]["content"]

        image_path = os.path.join(self.root_path, f"{image_id}")
        image = Image.open(image_path).convert("RGB")

        if crop != []:
            image = image.crop(tuple(crop))

        # for change detection
        if len(image_dirs) == 2:
            image_id2 = ann["images"][1]
            try:
                image_path = os.path.join(self.root_path, f"{image_id2}")
                image2 = Image.open(image_path).convert("RGB")

                if crop != []:
                    image2 = image2.crop(tuple(crop))

                img1 = np.array(image)
                img2 = np.array(image2)
                img = np.zeros(img1.shape)
                half1 = np.concatenate((img1, img), axis=0)
                half2 = np.concatenate((img, img2), axis=0)
                image = np.concatenate((half1, half2), axis=1)
                image = Image.fromarray(np.uint8(image))
            except:
                print("can not find/open {}".format(image_path))

        return image_dirs, prompt, label, image, crop


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Function to load JSON data from a given file path
def load_json_data(eval_file_path):
    with open(eval_file_path, "r") as file:
        json_data = json.load(file)
    return json_data


# Function to extract region of interest (ROI) bounding boxes using regex patterns
def extract_roi(input_string, pattern=r"\{<(\d+)><(\d+)><(\d+)><(\d+)>\|<(\d+)>"):
    # Regular expression pattern to capture the required groups
    pattern = pattern
    # Find all matches
    matches = re.findall(pattern, input_string)

    # Extract the values
    extracted_values = [match for match in matches]

    return extracted_values


# Function to extract polygons from model-generated text
def extract_polygons(generated_text, image_size, coordinates_quantizer):
    polygon_start_token = "<poly>"
    polygon_end_token = "</poly>"
    polygon_sep_token = "<sep>"
    with_box_at_start = False
    polygons_instance_pattern = (
        rf"{re.escape(polygon_start_token)}(.*?){re.escape(polygon_end_token)}"
    )
    polygons_instances_parsed = list(
        re.finditer(polygons_instance_pattern, generated_text)
    )

    box_pattern = rf"((?:<\d+>)+)(?:{re.escape(polygon_sep_token)}|$)"
    all_polygons = []
    for _polygons_instances_parsed in polygons_instances_parsed:
        instance = {}

        if isinstance(_polygons_instances_parsed, str):
            polygons_parsed = list(re.finditer(box_pattern, _polygons_instances_parsed))
        else:
            polygons_parsed = list(
                re.finditer(box_pattern, _polygons_instances_parsed.group(1))
            )
        if len(polygons_parsed) == 0:
            continue

        # a list of list (polygon)
        bbox = []
        polygons = []
        for _polygon_parsed in polygons_parsed:
            # group 1: whole <\d+>...</\d+>
            _polygon = _polygon_parsed.group(1)
            # parse into list of int
            _polygon = [
                int(_loc_parsed.group(1))
                for _loc_parsed in re.finditer(r"<(\d+)>", _polygon)
            ]
            if with_box_at_start and len(bbox) == 0:
                if len(_polygon) > 4:
                    # no valid bbox prediction
                    bbox = _polygon[:4]
                    _polygon = _polygon[4:]
                else:
                    bbox = [0, 0, 0, 0]
            # abandon last element if is not paired
            if len(_polygon) % 2 == 1:
                _polygon = _polygon[:-1]
            # reshape into (n, 2)
            _polygon = (
                coordinates_quantizer.dequantize(
                    torch.tensor(np.array(_polygon).reshape(-1, 2)), size=image_size
                )
                .reshape(-1)
                .tolist()
            )
            # reshape back
            polygons.append(_polygon)
        all_polygons.append(polygons)
    return all_polygons


def collate_fn(batch, processor):
    image_ids, prompts, labels, images, crops = zip(*batch)
    inputs = processor(
        text=list(prompts),
        images=list(images),
        return_tensors="pt",
        padding=True,
    )
    return image_ids, prompts, images, labels, inputs, crops


# Main evaluation function
def eval_model(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(111)

    model_path = args.model_path
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("load model successfully")
    model = model.to(device)
    model = model.eval()

    coordinates_quantizer = CoordinatesQuantizer(
        "floor",
        (1000, 1000),
    )

    evaluate_datasets = []
    evaluate_datasets.append(args.eval_file)

    result_path = args.result_path

    model_name = args.model_name
    print(model_name)

    save_path = os.path.join(result_path, model_name)
    # today = datetime.date.today().strftime("%y%m%d")
    # save_path = os.path.join(result_path, model_name + "-" + today)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for eval_file_path in evaluate_datasets:
        json_data = load_json_data(eval_file_path)
        val_dataset = RSDataset(args, json_data)

        batch_size = args.batch_size
        num_workers = args.num_workers

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=partial(collate_fn, processor=processor),
            num_workers=num_workers,
        )

        task = (
            json_data[0]["conversations"][0]["content"].split("]\n")[0].split("\n[")[-1]
        )
        dataset_name = json_data[0]["images"][0].split("/")[0]

        print("Inference for ", dataset_name)

        predict_results = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation"):
                image_ids, prompts, images, labels, inputs, crops = batch

                generated_ids = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    pixel_values=inputs["pixel_values"].to(device),
                    max_new_tokens=4096,
                    num_beams=3,
                    do_sample=False,
                )
                generated_texts_batch = processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )

                for image_id, prompt, image, label, generated_text, crop in zip(
                    image_ids, prompts, images, labels, generated_texts_batch, crops
                ):

                    if task in [
                        "REG_VG",
                        "REG_DET_HBB",
                        "REG_DET_OBB",
                        "PIX_SEG",
                        "PIX_CHG",
                    ]:
                        if task == "REG_DET_OBB":  # obb detection
                            pred_bboxes = extract_roi(
                                generated_text,
                                pattern=r"<(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)>",
                            )
                            ground_roi = extract_roi(
                                label,
                                pattern=r"<(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)>",
                            )
                            answer = []
                            for bbox in pred_bboxes:
                                answer.append(list(bbox))
                            label = []
                            for bbox in ground_roi:
                                label.append(list(bbox))
                        elif task in ["REG_VG", "REG_DET_HBB"]:  # hbb detection
                            ground_roi = extract_roi(
                                label, pattern=r"<(\d+)><(\d+)><(\d+)><(\d+)>"
                            )
                            pred_bboxes = extract_roi(
                                generated_text, pattern=r"<(\d+)><(\d+)><(\d+)><(\d+)>"
                            )
                            answer = []
                            for bbox in pred_bboxes:
                                answer.append(list(bbox))
                            label = []
                            for bbox in ground_roi:
                                label.append(list(bbox))
                        elif (
                            task == "PIX_SEG" or task == "PIX_CHG"
                        ):  # segmentation and change detection
                            gt_polygons = extract_polygons(
                                label,
                                (image.width, image.height),
                                coordinates_quantizer,
                            )
                            pred_polygons = extract_polygons(
                                generated_text,
                                (image.width, image.height),
                                coordinates_quantizer,
                            )
                            answer = pred_polygons
                            label = gt_polygons
                        else:
                            print("Unknown task ", task)

                    else:
                        answer = (
                            generated_text.replace("</s>", "")
                            .replace("<s>", "")
                            .replace("<pad>", "")
                            .replace("</pad>", "")
                        )

                    result = dict()
                    result["image"] = image_id
                    result["crop"] = crop
                    result["question"] = prompt
                    result["answer"] = answer
                    result["gt"] = label
                    predict_results.append(result)

        final_out = dict()
        final_out["info"] = {"task": task, "model": model_name, "dataset": dataset_name}
        final_out["data"] = predict_results

        save_file_name = eval_file_path.split("/")[-1]
        file_save_path = os.path.join(save_path, save_file_name)
        with open(file_save_path, "w") as f:
            f.write(json.dumps(final_out, indent=4))
        print("Save result successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="checkpoint path")
    parser.add_argument("--eval-file", type=str, help="the evaluation json file")
    parser.add_argument(
        "--dataset-path", type=str, help="root path of the evaluation data"
    )
    parser.add_argument(
        "--model-name", type=str, default="eval_model", help="specify the model name"
    )
    parser.add_argument("--result-path", type=str, default="./")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()

    eval_model(args)
