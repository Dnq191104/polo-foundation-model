"""
Embedding Module for Fashion Retrieval

Provides OpenCLIP-based image and text embedding with consistent preprocessing.
Optimized for CPU inference with batching support.
"""

import numpy as np
import torch
from PIL import Image
from typing import List, Union, Optional
from tqdm import tqdm
import open_clip


class OpenCLIPEmbedder:
    """
    Wrapper for OpenCLIP models to generate normalized embeddings.
    
    Features:
    - Consistent preprocessing for images and text
    - Batched encoding for efficiency
    - L2 normalization for cosine similarity via dot product
    - CPU-friendly with configurable device
    
    Args:
        model_name: OpenCLIP model name (default: 'ViT-B-32')
        pretrained: Pretrained weights (default: 'openai')
        device: Device to run on (default: 'cpu')
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None
    ):
        """Initialize the embedder with specified model."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Load model and preprocessing
        print(f"Loading OpenCLIP model: {model_name} ({pretrained})")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        with torch.no_grad():
            dummy_image = torch.zeros(1, 3, 224, 224).to(self.device)
            dummy_features = self.model.encode_image(dummy_image)
            self.embedding_dim = dummy_features.shape[-1]
        
        print(f"Model loaded on {self.device}, embedding dim: {self.embedding_dim}")
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-12)
        return embeddings / norms
    
    def encode_image(
        self,
        image: Union[Image.Image, List[Image.Image]],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode image(s) to embedding vector(s).
        
        Args:
            image: Single PIL Image or list of PIL Images
            normalize: L2 normalize output (default: True)
            show_progress: Show progress bar for lists
            
        Returns:
            Normalized embedding(s), shape (D,) or (N, D)
        """
        single_input = not isinstance(image, list)
        if single_input:
            image = [image]
        
        embeddings = []
        
        with torch.no_grad():
            iterator = tqdm(image, desc="Encoding images") if show_progress else image
            
            for img in iterator:
                # Preprocess and add batch dimension
                img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                features = self.model.encode_image(img_tensor)
                embeddings.append(features.cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        
        if normalize:
            embeddings = self._normalize(embeddings)
        
        return embeddings[0] if single_input else embeddings
    
    def encode_image_batch(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode images in batches for efficiency.
        
        Args:
            images: List of PIL Images
            batch_size: Batch size for encoding
            normalize: L2 normalize output
            show_progress: Show progress bar
            
        Returns:
            Normalized embeddings, shape (N, D)
        """
        embeddings = []
        
        with torch.no_grad():
            num_batches = (len(images) + batch_size - 1) // batch_size
            iterator = range(num_batches)
            if show_progress:
                iterator = tqdm(iterator, desc="Encoding image batches")
            
            for i in iterator:
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(images))
                batch_images = images[start_idx:end_idx]
                
                # Preprocess batch
                batch_tensors = torch.stack([
                    self.preprocess(img) for img in batch_images
                ]).to(self.device)
                
                features = self.model.encode_image(batch_tensors)
                embeddings.append(features.cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        
        if normalize:
            embeddings = self._normalize(embeddings)
        
        return embeddings
    
    def encode_text(
        self,
        text: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) to embedding vector(s).
        
        Args:
            text: Single string or list of strings
            normalize: L2 normalize output (default: True)
            show_progress: Show progress bar for lists
            
        Returns:
            Normalized embedding(s), shape (D,) or (N, D)
        """
        single_input = not isinstance(text, list)
        if single_input:
            text = [text]
        
        embeddings = []
        
        with torch.no_grad():
            iterator = tqdm(text, desc="Encoding text") if show_progress else text
            
            for t in iterator:
                tokens = self.tokenizer([t]).to(self.device)
                features = self.model.encode_text(tokens)
                embeddings.append(features.cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        
        if normalize:
            embeddings = self._normalize(embeddings)
        
        return embeddings[0] if single_input else embeddings
    
    def encode_text_batch(
        self,
        texts: List[str],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts in batches for efficiency.
        
        Args:
            texts: List of strings
            batch_size: Batch size for encoding
            normalize: L2 normalize output
            show_progress: Show progress bar
            
        Returns:
            Normalized embeddings, shape (N, D)
        """
        embeddings = []
        
        with torch.no_grad():
            num_batches = (len(texts) + batch_size - 1) // batch_size
            iterator = range(num_batches)
            if show_progress:
                iterator = tqdm(iterator, desc="Encoding text batches")
            
            for i in iterator:
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]
                
                # Tokenize batch
                tokens = self.tokenizer(batch_texts).to(self.device)
                features = self.model.encode_text(tokens)
                embeddings.append(features.cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        
        if normalize:
            embeddings = self._normalize(embeddings)
        
        return embeddings
    
    def get_model_info(self) -> dict:
        """Get model configuration information."""
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
        }

