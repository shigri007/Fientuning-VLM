import numpy as np
import supervision as sv
from config import DEVICE


def collect_predictions_and_targets(model, processor, dataset, classes):
    targets = []
    predictions = []

    for i in range(len(dataset.dataset)):
        image, data = dataset.dataset[i]
        prefix = data["prefix"]
        suffix = data["suffix"]

        inputs = processor(text=prefix, images=image, return_tensors="pt").to(DEVICE)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        prediction = processor.post_process_generation(generated_text, task="<OD>", image_size=image.size)
        prediction = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, prediction, resolution_wh=image.size)

        prediction = prediction[np.isin(prediction["class_name"], classes)]
        prediction.class_id = np.array(
            [classes.index(class_name) if class_name in classes else -1 for class_name in prediction["class_name"]]
        )
        prediction.confidence = np.ones(len(prediction))

        target = processor.post_process_generation(suffix, task="<OD>", image_size=image.size)
        target = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, target, resolution_wh=image.size)
        target.class_id = np.array(
            [classes.index(class_name) if class_name in classes else -1 for class_name in target["class_name"]]
        )

        targets.append(target)
        predictions.append(prediction)

    return predictions, targets


def render_inference_results(model, dataset, count):
    count = min(count, len(dataset))
    for i in range(count):
        image, data = dataset.dataset[i]
        prefix = data["prefix"]
        inputs = processor(text=prefix, images=image, return_tensors="pt").to(DEVICE)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        print(f"Example {i + 1}: {generated_text}")
