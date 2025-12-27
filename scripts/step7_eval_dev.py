#!/usr/bin/env python
"""
Fast Dev Evaluation for Step 7

Quick evaluation on small stratified sample for iteration speed.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.protocol_model import ProtocolModel


def load_dev_ids(dev_ids_path):
    """Load dev evaluation IDs"""
    with open(dev_ids_path, 'r') as f:
        return json.load(f)


def dev_evaluate(model, dataset, dev_ids, device='cuda', n_queries=100):
    """Fast dev evaluation on stratified sample"""

    model.eval()
    dev_ids = dev_ids[:n_queries]  # Limit for speed

    print(f"Dev evaluating on {len(dev_ids)} queries...")

    # Compute all embeddings (cached)
    if not hasattr(dev_evaluate, '_embeddings'):
        print("Computing dataset embeddings...")
        all_embeddings = []

        for i in tqdm(range(0, len(dataset), 64), desc="Embedding"):
            batch_items = [dataset[j] for j in range(i, min(i+64, len(dataset)))]
            batch_images = [item['image'] for item in batch_items]
            batch_tensors = torch.stack([model.preprocess(img) for img in batch_images]).to(device)

            with torch.no_grad():
                embeddings = model.forward_image(batch_tensors, return_attributes=False)['embedding']
            all_embeddings.append(embeddings.cpu())

        dev_evaluate._embeddings = torch.cat(all_embeddings)
        dev_evaluate._embeddings_norm = dev_evaluate._embeddings / dev_evaluate._embeddings.norm(dim=1, keepdim=True)

    embeddings_norm = dev_evaluate._embeddings_norm

    # Evaluate queries
    total_correct = 0
    category_correct = 0

    for query_idx in dev_ids:
        query_item = dataset[query_idx]
        query_category = query_item.get('category2', '')

        # Get query embedding
        query_img = query_item['image']
        query_tensor = model.preprocess(query_img).unsqueeze(0).to(device)

        with torch.no_grad():
            query_emb = model.forward_image(query_tensor, return_attributes=False)['embedding']
        query_emb = query_emb / query_emb.norm(dim=1, keepdim=True)
        query_emb = query_emb.cpu()

        # Compute similarities
        similarities = (query_emb @ embeddings_norm.T).squeeze()
        similarities[query_idx] = -float('inf')  # Exclude self

        # Top-10 evaluation
        top10_indices = torch.topk(similarities, 10).indices.tolist()
        top10_items = [dataset[idx] for idx in top10_indices]

        # Category accuracy
        if any(item.get('category2') == query_category for item in top10_items):
            category_correct += 1
            total_correct += 1

    return {
        'total_acc': total_correct / len(dev_ids),
        'category_acc': category_correct / len(dev_ids),
        'n_queries': len(dev_ids)
    }


def main():
    parser = argparse.ArgumentParser(description="Fast dev evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path")
    parser.add_argument("--dev_ids", type=str, required=True, help="Dev eval IDs JSON")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--n_queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--device", type=str, default=None, help="Device")

    args = parser.parse_args()

    # Load checkpoint
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create model and load weights
    model = ProtocolModel(
        model_name="ViT-B-32",
        pretrained="openai",
        projection_hidden=None,
        use_attribute_heads=True,
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load dataset
    dataset = load_from_disk(args.dataset)['validation']

    # Load dev IDs
    dev_ids = load_dev_ids(args.dev_ids)

    # Run evaluation
    results = dev_evaluate(model, dataset, dev_ids, device, args.n_queries)

    print("Dev Evaluation Results:")
    print(".3f")
    print(f"Queries evaluated: {results['n_queries']}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
