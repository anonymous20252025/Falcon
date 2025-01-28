import argparse
import re
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


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


# Function to extract region of interest (ROI) bounding boxes using regex patterns
def extract_roi(input_string, pattern=r"\{<(\d+)><(\d+)><(\d+)><(\d+)>\|<(\d+)>"):
    # Regular expression pattern to capture the required groups
    pattern = pattern
    # Find all matches
    matches = re.findall(pattern, input_string)

    # Extract the values
    extracted_values = [match for match in matches]

    return extracted_values


def infrence_on_single_image(args):
    coordinates_quantizer = CoordinatesQuantizer(
        "floor",
        (1000, 1000),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
    )

    image = Image.open(args.image_path).convert("RGB")

    if args.image2_path is not None:
        image2 = Image.open(args.image2_path).convert("RGB")
        img1 = np.array(image)
        img2 = np.array(image2)
        img = np.zeros(img1.shape)
        half1 = np.concatenate((img1, img), axis=0)
        half2 = np.concatenate((img, img2), axis=0)
        image = np.concatenate((half1, half2), axis=1)
        image = Image.fromarray(np.uint8(image))
    inputs = processor(text=args.prompt, images=image, return_tensors="pt")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=8192,
        num_beams=3,
        do_sample=False,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    if args.post_process_type in [
        "REG_VG",
        "REG_DET_HBB",
        "REG_DET_OBB",
        "PIX_SEG",
        'PIX_CHG'
    ]:
        if args.post_process_type == "REG_DET_OBB":     # obb detection
            pred_bboxes = extract_roi(
                generated_text,
                pattern=r"<(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)>",
            )
            answer = []
            for bbox in pred_bboxes:
                answer.append(list(bbox))
        elif args.post_process_type in ["REG_VG", "REG_DET_HBB"]:  # hbb detection
            pred_bboxes = extract_roi(
                generated_text, pattern=r"<(\d+)><(\d+)><(\d+)><(\d+)>"
            )
            answer = []
            for bbox in pred_bboxes:
                answer.append(list(bbox))
        elif args.post_process_type == "PIX_SEG" or args.post_process_type == 'PIX_CHG':  # segmentation and change detection
            pred_polygons = extract_polygons(
                generated_text, (image.width, image.height), coordinates_quantizer
            )
            answer = pred_polygons
        else:
            print("Unknown task ", args.post_process_type)

    else:
        answer = generated_text.replace("</s>", "").replace("<s>", "").replace('<pad>','').replace('</pad>','')
    # parsed_answer = processor.post_process_generation(
    #     generated_text, task=args.post_process_type +"\n", image_size=(image.width, image.height)
    # )
    print('User:', args.prompt)
    print('Falcon:', answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="checkpoint path")
    parser.add_argument("--image_path", type=str, required=True, help="path of the image sample")
    parser.add_argument("--image2_path", type=str, default=None, help="path of the second image sample")
    parser.add_argument("--post_process_type", type=str, required=True, help="post process type for raw output of model")
    parser.add_argument("--prompt", type=str, required=True, help="task prompt for Falcon")

    args = parser.parse_args()

    infrence_on_single_image(args)