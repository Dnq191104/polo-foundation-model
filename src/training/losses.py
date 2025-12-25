"""
Training Losses for Step 7

Contrastive retrieval loss + attribute classification losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for retrieval.
    
    Pulls positives closer and pushes negatives apart.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature scaling parameter
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            anchor_embeddings: Anchor embeddings (B, D)
            positive_embeddings: Positive embeddings (B, D)
            negative_embeddings: Optional explicit negatives (N, D)
            
        Returns:
            Loss scalar
        """
        batch_size = anchor_embeddings.size(0)
        
        # Normalize
        anchor_embeddings = F.normalize(anchor_embeddings, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, dim=1)
        
        # Positive similarity
        pos_sim = (anchor_embeddings * positive_embeddings).sum(dim=1) / self.temperature
        
        # Negative similarities (in-batch)
        # Treat all other items in batch as negatives
        all_embeddings = torch.cat([positive_embeddings, anchor_embeddings], dim=0)
        neg_sim = anchor_embeddings @ all_embeddings.t() / self.temperature
        
        # Remove self-similarity
        mask = torch.eye(batch_size, device=anchor_embeddings.device)
        neg_sim = neg_sim[:, :batch_size] * (1 - mask) + neg_sim[:, batch_size:]
        
        # If explicit negatives provided, add them
        if negative_embeddings is not None:
            negative_embeddings = F.normalize(negative_embeddings, dim=1)
            explicit_neg_sim = anchor_embeddings @ negative_embeddings.t() / self.temperature
            neg_sim = torch.cat([neg_sim, explicit_neg_sim], dim=1)
        
        # LogSumExp trick for numerical stability
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor_embeddings.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


class AttributeLoss(nn.Module):
    """
    Multi-attribute classification loss.
    
    Predicts material, pattern, neckline, sleeve from embeddings.
    """
    
    def __init__(
        self,
        attribute_weights: Optional[Dict[str, float]] = None,
        ignore_index: int = -1
    ):
        """
        Initialize attribute loss.
        
        Args:
            attribute_weights: Per-attribute loss weights
            ignore_index: Index to ignore (e.g., for unknown labels)
        """
        super().__init__()
        
        default_weights = {
            'material': 1.0,
            'pattern': 1.0,
            'neckline': 2.0,  # Higher weight for neckline!
            'sleeve': 1.0,
        }
        
        self.attribute_weights = attribute_weights or default_weights
        self.ignore_index = ignore_index
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute attribute losses.
        
        Args:
            predictions: Dict of attribute_name -> logits (B, num_classes)
            labels: Dict of attribute_name -> labels (B,)
            
        Returns:
            Dict with 'total' and per-attribute losses
        """
        losses = {}
        total_loss = 0.0
        
        for attr_name, logits in predictions.items():
            if attr_name not in labels:
                continue
            
            target = labels[attr_name]
            weight = self.attribute_weights.get(attr_name, 1.0)
            
            # Skip if all labels are ignore_index
            if (target == self.ignore_index).all():
                continue
            
            loss = F.cross_entropy(
                logits,
                target,
                ignore_index=self.ignore_index
            )
            
            losses[attr_name] = loss
            total_loss += weight * loss
        
        losses['total'] = total_loss
        
        return losses


class CombinedLoss(nn.Module):
    """
    Combined loss: Contrastive + Attribute.
    
    Main objective for Step 7 training.
    """
    
    def __init__(
        self,
        contrastive_weight: float = 1.0,
        attribute_weight: float = 0.3,
        temperature: float = 0.07,
        attribute_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize combined loss.
        
        Args:
            contrastive_weight: Weight for contrastive loss
            attribute_weight: Weight for attribute loss
            temperature: Temperature for InfoNCE
            attribute_weights: Per-attribute weights
        """
        super().__init__()
        
        self.contrastive_weight = contrastive_weight
        self.attribute_weight = attribute_weight
        
        self.contrastive_loss = InfoNCELoss(temperature=temperature)
        self.attribute_loss = AttributeLoss(attribute_weights=attribute_weights)
    
    def forward(
        self,
        anchor_output: Dict[str, torch.Tensor],
        positive_output: Dict[str, torch.Tensor],
        attribute_labels: Optional[Dict[str, torch.Tensor]] = None,
        negative_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            anchor_output: Anchor forward output (embedding + attributes)
            positive_output: Positive forward output
            attribute_labels: Ground truth attribute labels
            negative_embeddings: Optional explicit negative embeddings
            
        Returns:
            Dict with 'total' and component losses
        """
        losses = {}
        
        # Contrastive loss
        contrastive = self.contrastive_loss(
            anchor_output['embedding'],
            positive_output['embedding'],
            negative_embeddings
        )
        losses['contrastive'] = contrastive
        
        # Attribute loss (if attributes predicted and labels provided)
        if attribute_labels and 'attributes' in anchor_output:
            attr_losses = self.attribute_loss(
                anchor_output['attributes'],
                attribute_labels
            )
            
            losses['attribute_total'] = attr_losses['total']
            
            for attr_name, loss_val in attr_losses.items():
                if attr_name != 'total':
                    losses[f'attribute_{attr_name}'] = loss_val
        else:
            losses['attribute_total'] = torch.tensor(0.0, device=anchor_output['embedding'].device)
        
        # Combined
        total = (
            self.contrastive_weight * losses['contrastive'] +
            self.attribute_weight * losses['attribute_total']
        )
        
        losses['total'] = total
        
        return losses


def create_combined_loss(
    contrastive_weight: float = 1.0,
    attribute_weight: float = 0.3,
    neckline_weight: float = 2.0,
    temperature: float = 0.07
) -> CombinedLoss:
    """
    Factory function to create CombinedLoss.
    
    Args:
        contrastive_weight: Contrastive loss weight
        attribute_weight: Attribute loss weight
        neckline_weight: Extra weight for neckline attribute
        temperature: Temperature for contrastive loss
        
    Returns:
        Configured CombinedLoss instance
    """
    attribute_weights = {
        'material': 1.0,
        'pattern': 1.0,
        'neckline': neckline_weight,  # Emphasize neckline!
        'sleeve': 1.0,
    }
    
    return CombinedLoss(
        contrastive_weight=contrastive_weight,
        attribute_weight=attribute_weight,
        temperature=temperature,
        attribute_weights=attribute_weights
    )

