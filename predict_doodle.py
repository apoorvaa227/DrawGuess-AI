import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


def main():
    parser = argparse.ArgumentParser(description="Predict doodle class from image")
    parser.add_argument("--image", type=str, required=True, help="Path to doodle image")
    parser.add_argument("--model-path", type=str, default="runs/doodle_classifier/best_model.pt")
    parser.add_argument("--labels-path", type=str, default="runs/doodle_classifier/labels.json")
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(args.labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    tf = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ]
    )

    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)

    topk = min(args.topk, len(labels))
    values, indices = torch.topk(probs, k=topk)

    print(f"Prediction for: {image_path}")
    print("Top classes:")
    for rank, (v, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        print(f"{rank}. {labels[idx]} ({v * 100:.2f}%)")


if __name__ == "__main__":
    main()
