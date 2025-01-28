from PIL import Image
from torch.utils.data import Dataset
import json
import numpy as np
import os

Image.MAX_IMAGE_PIXELS = None


class BaseDataset(Dataset):
    def __init__(self, split):
        self._split = split
        self.name = "BaseDataset"
        self.data = []
        self.task_prompt = ""

    def __len__(self):
        return len(self.data)

    def correct_casing_finqa(self, text, is_question=False):
        if text and text[0].islower():
            text = text.capitalize()
        if not text.endswith(".") and not is_question:
            text += "."
        if not text.endswith("?") and is_question:
            text += "?"
        return text


class RSDataset(BaseDataset):
    def __init__(self, split, json_file):
        super().__init__(split)
        json_file = json_file
        self.name = "RS"
        with open(json_file) as file:
            self.data = json.load(file)
        self.task_prompt = ""

    def __getitem__(self, idx):
        example = self.data[idx]
        question = example["conversations"][0]["content"].replace("<image>\n", "")
        answers = example["conversations"][1]["content"]
        img_dir = example["images"]

        if "crop" in example.keys():
            img_crop = example["crop"]
        else:
            img_crop = []

        if img_dir is None:
            image = Image.fromarray(np.zeros((448, 448, 3)), mode="RGB")
        else:
            image_dir = img_dir[0]
            try:
                image = Image.open(image_dir).convert("RGB")
                if img_crop != []:
                    image = image.crop(img_crop)
            except:
                print("can not find/open {}".format(image_dir))
                image = Image.fromarray(np.zeros((448, 448, 3)), mode="RGB")

            if len(img_dir) == 2:
                image_dir = img_dir[1]
                try:
                    image2 = Image.open(image_dir).convert("RGB")
                    if img_crop != []:
                        image2 = image2.crop(img_crop)

                    img1 = np.array(image)
                    img2 = np.array(image2)
                    img = np.zeros(img1.shape)
                    half1 = np.concatenate((img1, img), axis=0)
                    half2 = np.concatenate((img, img2), axis=0)
                    image = np.concatenate((half1, half2), axis=1)
                    image = Image.fromarray(np.uint8(image))

                except:
                    print("can not find/open {}".format(image_dir))

        return question, answers, image
