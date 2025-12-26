"""
Protocol Model for Step 7

Frozen OpenCLIP backbone with trainable projection heads and attribute heads.
CPU-friendly with minimal trainable parameters.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

import open_clip
from PIL import Image


class ProjectionHead(nn.Module):
    """
    Lightweight projection head for embeddings.
    
    Can be linear or small MLP.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize projection head.
        
        Args:
            input_dim: Input embedding dimension
            output_dim: Output embedding dimension
            hidden_dim: Hidden dimension for MLP (None for linear)
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_dim is None:
            # Linear projection
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            # MLP projection
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project embeddings."""
        return self.projection(x)


class AttributeHead(nn.Module):
    """
    Attribute classification head.
    
    Predicts single attribute from embedding.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None
    ):
        """
        Initialize attribute head.
        
        Args:
            input_dim: Input embedding dimension
            num_classes: Number of attribute classes
            hidden_dim: Optional hidden dimension
        """
        super().__init__()
        
        if hidden_dim is None:
            self.classifier = nn.Linear(input_dim, num_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify attribute."""
        return self.classifier(x)


class ProtocolModel(nn.Module):
    """
    Protocol model: Frozen OpenCLIP + trainable adapters.
    
    Architecture:
    - Frozen OpenCLIP ViT-B/32 (or specified model)
    - Trainable image projection head
    - Trainable text projection head
    - Optional attribute heads (material, pattern, neckline, sleeve)
    - Optional learnable fusion weight
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        projection_dim: Optional[int] = None,
        projection_hidden: Optional[int] = None,
        use_attribute_heads: bool = True,
        attribute_dims: Optional[Dict[str, int]] = None,
        learn_fusion_weight: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize protocol model.
        
        Args:
            model_name: OpenCLIP model name
            pretrained: Pretrained weights
            projection_dim: Output dimension (None = same as backbone)
            projection_hidden: Hidden dim for projection MLP
            use_attribute_heads: Whether to add attribute heads
            attribute_dims: Dict of attribute_name -> num_classes
            learn_fusion_weight: Learn fusion weight (vs fixed 0.7)
            device: Device to use
        """
        super().__init__()
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Load frozen OpenCLIP
        print(f"Loading OpenCLIP model: {model_name} ({pretrained})")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Get embedding dimension
        with torch.no_grad():
            dummy_image = torch.zeros(1, 3, 224, 224).to(self.device)
            dummy_features = self.clip_model.encode_image(dummy_image)
            self.clip_dim = dummy_features.shape[-1]
        
        self.projection_dim = projection_dim or self.clip_dim
        
        # Trainable projection heads
        self.img_projection = ProjectionHead(
            self.clip_dim,
            self.projection_dim,
            hidden_dim=projection_hidden
        ).to(self.device)
        
        self.txt_projection = ProjectionHead(
            self.clip_dim,
            self.projection_dim,
            hidden_dim=projection_hidden
        ).to(self.device)
        
        # Fusion weight
        self.learn_fusion_weight = learn_fusion_weight
        if learn_fusion_weight:
            self.fusion_weight = nn.Parameter(torch.tensor(0.7))
        else:
            self.register_buffer('fusion_weight', torch.tensor(0.7))
        
        # Attribute heads
        self.use_attribute_heads = use_attribute_heads
        self.attribute_heads = nn.ModuleDict()
        
        if use_attribute_heads:
            default_dims = {
                'category': 25,  # Adjust based on your schema
                'material': 20,  # Adjust based on your schema
                'pattern': 15,
                'neckline': 12,
                'sleeve': 10,
            }
            attribute_dims = attribute_dims or default_dims
            
            for attr_name, num_classes in attribute_dims.items():
                self.attribute_heads[attr_name] = AttributeHead(
                    self.projection_dim,
                    num_classes
                ).to(self.device)
        
        print(f"Protocol model initialized on {self.device}")
        print(f"  CLIP dim: {self.clip_dim}")
        print(f"  Projection dim: {self.projection_dim}")
        print(f"  Trainable params: {self.count_trainable_parameters():,}")
    
    def count_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward_image(self, images: torch.Tensor, return_attributes: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for images.
        
        Args:
            images: Image tensor (B, 3, H, W)
            return_attributes: Whether to return attribute predictions
            
        Returns:
            Dict with 'embedding' and optionally 'attributes'
        """
        # Extract CLIP features (frozen)
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(images)
            clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        
        # Project (trainable)
        embeddings = self.img_projection(clip_features)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        result = {'embedding': embeddings}
        
        # Attribute predictions
        if return_attributes and self.use_attribute_heads:
            attributes = {}
            for attr_name, head in self.attribute_heads.items():
                attributes[attr_name] = head(embeddings)
            result['attributes'] = attributes
        
        return result
    
    def forward_text(self, texts: List[str], return_attributes: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for text.
        
        Args:
            texts: List of text strings
            return_attributes: Whether to return attribute predictions
            
        Returns:
            Dict with 'embedding' and optionally 'attributes'
        """
        # Tokenize and extract CLIP features (frozen)
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            clip_features = self.clip_model.encode_text(tokens)
            clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        
        # Project (trainable)
        embeddings = self.txt_projection(clip_features)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        result = {'embedding': embeddings}
        
        # Attribute predictions
        if return_attributes and self.use_attribute_heads:
            attributes = {}
            for attr_name, head in self.attribute_heads.items():
                attributes[attr_name] = head(embeddings)
            result['attributes'] = attributes
        
        return result
    
    def compute_fused_similarity(
        self,
        img_embeddings: torch.Tensor,
        txt_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute fused similarity scores.
        
        Args:
            img_embeddings: Image embeddings (B, D)
            txt_embeddings: Text embeddings (B, D)
            
        Returns:
            Fused similarity matrix (B, B)
        """
        # Image similarity
        img_sim = img_embeddings @ img_embeddings.t()
        
        # Text similarity
        txt_sim = txt_embeddings @ txt_embeddings.t()
        
        # Fused
        w = self.fusion_weight.clamp(0, 1)  # Keep in [0, 1]
        fused_sim = w * img_sim + (1 - w) * txt_sim
        
        return fused_sim
    
    @torch.no_grad()
    def encode_image_numpy(self, image: Image.Image, normalize: bool = True) -> np.ndarray:
        """
        Encode single image to numpy array (for inference).
        
        Args:
            image: PIL Image
            normalize: L2 normalize output
            
        Returns:
            Embedding as numpy array
        """
        self.eval()
        
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        result = self.forward_image(img_tensor)
        embedding = result['embedding'].cpu().numpy()[0]
        
        if normalize:
            embedding = embedding / (np.linalg.norm(embedding) + 1e-12)
        
        return embedding
    
    @torch.no_grad()
    def encode_text_numpy(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encode single text to numpy array (for inference).
        
        Args:
            text: Text string
            normalize: L2 normalize output
            
        Returns:
            Embedding as numpy array
        """
        self.eval()
        
        result = self.forward_text([text])
        embedding = result['embedding'].cpu().numpy()[0]
        
        if normalize:
            embedding = embedding / (np.linalg.norm(embedding) + 1e-12)
        
        return embedding
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Optional[Dict] = None):
        """
        Save model checkpoint.
        
        Args:
            path: Output path
            epoch: Current epoch
            optimizer_state: Optional optimizer state dict
        """
        checkpoint = {
            'epoch': epoch,
            'model_name': self.model_name,
            'pretrained': self.pretrained,
            'projection_dim': self.projection_dim,
            'clip_dim': self.clip_dim,
            'img_projection_state': self.img_projection.state_dict(),
            'txt_projection_state': self.txt_projection.state_dict(),
            'fusion_weight': self.fusion_weight.item(),
            'learn_fusion_weight': self.learn_fusion_weight,
            'use_attribute_heads': self.use_attribute_heads,
        }
        
        if self.use_attribute_heads:
            checkpoint['attribute_heads_state'] = {
                name: head.state_dict()
                for name, head in self.attribute_heads.items()
            }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state'] = optimizer_state
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[str] = None):
        """
        Load model from checkpoint.
        
        Args:
            path: Checkpoint path
            device: Device to load to
            
        Returns:
            Loaded ProtocolModel instance
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # Create model with same config
        model = cls(
            model_name=checkpoint['model_name'],
            pretrained=checkpoint['pretrained'],
            projection_dim=checkpoint['projection_dim'],
            learn_fusion_weight=checkpoint['learn_fusion_weight'],
            use_attribute_heads=checkpoint['use_attribute_heads'],
            device=device
        )
        
        # Load trainable weights (fail loudly on architecture mismatch)
        try:
            model.img_projection.load_state_dict(checkpoint['img_projection_state'])
            print("Loaded img_projection weights")
        except RuntimeError as e:
            raise ValueError(
                f"Failed to load img_projection weights. "
                f"Architecture mismatch: {e}\n"
                f"Checkpoint was saved with a different projection_hidden setting."
            )

        try:
            model.txt_projection.load_state_dict(checkpoint['txt_projection_state'])
            print("Loaded txt_projection weights")
        except RuntimeError as e:
            raise ValueError(
                f"Failed to load txt_projection weights. "
                f"Architecture mismatch: {e}\n"
                f"Checkpoint was saved with a different projection_hidden setting."
            )
        
        if model.learn_fusion_weight:
            model.fusion_weight.data = torch.tensor(checkpoint['fusion_weight'])
        
        if model.use_attribute_heads and 'attribute_heads_state' in checkpoint:
            for name, state_dict in checkpoint['attribute_heads_state'].items():
                if name in model.attribute_heads:
                    model.attribute_heads[name].load_state_dict(state_dict)
        
        model.to(model.device)
        
        return model


def create_protocol_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    projection_hidden: Optional[int] = 256,
    use_attribute_heads: bool = True,
    device: Optional[str] = None
) -> ProtocolModel:
    """
    Factory function to create ProtocolModel.
    
    Args:
        model_name: OpenCLIP model name
        pretrained: Pretrained weights
        projection_hidden: Hidden dimension for projection MLP
        use_attribute_heads: Whether to use attribute heads
        device: Device
        
    Returns:
        Configured ProtocolModel instance
    """
    return ProtocolModel(
        model_name=model_name,
        pretrained=pretrained,
        projection_hidden=projection_hidden,
        use_attribute_heads=use_attribute_heads,
        device=device
    )

