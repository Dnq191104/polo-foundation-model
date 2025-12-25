"""
Fashion Attribute Extraction Module

Implements rule-based extraction of structured fashion attributes from product
text descriptions. Supports material, pattern, neckline, and sleeve attributes.

Key Features:
- Sentence-level parsing to associate attributes with specific garments
- Category-aware extraction (only extracts from sentences about the target garment)
- Attribute compatibility filtering (shorts don't have neckline/sleeve)
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set


class AttributeExtractor:
    """
    Extracts structured fashion attributes from product text descriptions.
    
    Uses sentence-level parsing to correctly associate attributes with the
    target garment (identified by category2), avoiding contamination from
    secondary garments mentioned in the same text.
    
    Attributes:
        schema: Loaded attribute schema with triggers and rules
        config: Extraction configuration
        garment_keywords: Keywords to identify garment types in text
    """
    
    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize the extractor with a schema file.
        
        Args:
            schema_path: Path to YAML schema file. If None, auto-detects.
        """
        if schema_path is None:
            schema_path = self._auto_detect_schema_path()
        
        self.schema_path = Path(schema_path)
        self._load_schema()
        self._compile_patterns()
    
    def _auto_detect_schema_path(self) -> str:
        """Auto-detect the schema path based on current working directory."""
        current_dir = Path.cwd()
        
        # Check various possible locations
        candidates = [
            current_dir / "src" / "datasets" / "attribute_schema.yaml",
            current_dir.parent / "src" / "datasets" / "attribute_schema.yaml",
            Path(__file__).parent / "attribute_schema.yaml",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        
        # Default fallback
        return "src/datasets/attribute_schema.yaml"
    
    def _load_schema(self) -> None:
        """Load the YAML schema file."""
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            self._full_schema = yaml.safe_load(f)
        
        # Extract config
        self.config = self._full_schema.get('config', {})
        self.mutually_exclusive = set(self.config.get('mutually_exclusive', ['sleeve', 'neckline']))
        self.multi_value = set(self.config.get('multi_value', ['material', 'pattern']))
        self.unknown_value = self.config.get('unknown_value', 'unknown')
        
        # Extract attribute compatibility rules
        self.attribute_compatibility = self._full_schema.get('attribute_compatibility', {
            'upper': ['material', 'pattern', 'neckline', 'sleeve'],
            'lower': ['material', 'pattern'],
            'full': ['material', 'pattern', 'neckline', 'sleeve'],
        })
        
        # Extract garment keywords for sentence identification
        self.garment_keywords = self._full_schema.get('garment_keywords', {})
        
        # Build keyword → scope mapping
        self._keyword_to_scope: Dict[str, str] = {}
        for scope, keywords in self.garment_keywords.items():
            for keyword in keywords:
                self._keyword_to_scope[keyword.lower()] = scope
        
        # Extract noise patterns
        noise_config = self._full_schema.get('noise_patterns', {})
        self.noise_phrases = []
        for category, phrases in noise_config.items():
            self.noise_phrases.extend(phrases)
        
        # Extract garment scopes (category2 → scope mapping)
        self.garment_scopes = self._full_schema.get('garment_scopes', {})
        
        # Build category → scope mapping
        self._category_to_scope = {}
        for scope, categories in self.garment_scopes.items():
            for cat in categories:
                self._category_to_scope[cat.lower()] = scope
        
        # Extract attribute schemas (material, pattern, neckline, sleeve)
        self.attributes = {}
        skip_keys = {'config', 'noise_patterns', 'garment_scopes', 
                     'attribute_compatibility', 'garment_keywords'}
        for attr_name, attr_config in self._full_schema.items():
            if attr_name not in skip_keys and isinstance(attr_config, dict):
                self.attributes[attr_name] = attr_config
    
    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficient matching."""
        self._trigger_patterns: Dict[str, Dict[str, List[re.Pattern]]] = {}
        
        for attr_name, attr_config in self.attributes.items():
            self._trigger_patterns[attr_name] = {}
            
            for tag_name, tag_config in attr_config.items():
                patterns = []
                for trigger in tag_config.get('triggers', []):
                    # Create word-boundary pattern for exact matching
                    trigger_lower = trigger.lower()
                    if ' ' in trigger_lower:
                        # Allow flexible whitespace for compound terms
                        pattern_str = r'\b' + re.escape(trigger_lower).replace(r'\ ', r'\s+') + r'\b'
                    else:
                        pattern_str = r'\b' + re.escape(trigger_lower) + r'\b'
                    
                    try:
                        patterns.append(re.compile(pattern_str, re.IGNORECASE))
                    except re.error:
                        # Fallback to simple containment check
                        patterns.append(re.compile(re.escape(trigger_lower), re.IGNORECASE))
                
                self._trigger_patterns[attr_name][tag_name] = patterns
        
        # Compile garment keyword patterns
        self._garment_patterns: Dict[str, List[re.Pattern]] = {}
        for scope, keywords in self.garment_keywords.items():
            patterns = []
            for keyword in keywords:
                pattern_str = r'\b' + re.escape(keyword.lower()) + r'\b'
                try:
                    patterns.append(re.compile(pattern_str, re.IGNORECASE))
                except re.error:
                    pass
            self._garment_patterns[scope] = patterns
    
    def get_garment_scope(self, category2: str) -> str:
        """
        Determine garment scope from category2.

        Args:
            category2: Product category (e.g., 'blouses', 'pants', 'shorts')

        Returns:
            Scope: 'upper', 'lower', or 'full'
        """
        return self._category_to_scope.get(category2.lower(), 'upper')

    def _anchor_attribute_to_subject(
        self,
        sentence: str,
        target_category: str
    ) -> List[Tuple[str, str, str, int]]:
        """
        Anchor attributes to specific garments using pattern matching.

        Looks for patterns like:
        - "[garment] is [material]"
        - "[garment] has [attribute]"
        - "The [attribute] of [garment] is [value]"
        - "[garment] with [attribute]"

        Returns list of (garment_mention, attribute_value, attribute_type, confidence_score)
        where garment_mention is the detected garment that matches target_category.
        """
        sentence_lower = sentence.lower()
        anchored_attributes = []

        # Get the correct scope and keywords for the target category
        target_scope = self.get_garment_scope(target_category)
        target_keywords = self.garment_keywords.get(target_scope, [])

        # Define anchoring patterns (restrictive to avoid over-matching)
        anchoring_patterns = [
            # Pattern: "[garment] is [material]" (single word for materials)
            r'\b(' + '|'.join([re.escape(kw) for kw in target_keywords]) + r')\b.*?is\s+([a-zA-Z]+)(?!\s+[a-zA-Z])',
            # Pattern: "[garment] has [attribute]" (1-2 words for attributes)
            r'\b(' + '|'.join([re.escape(kw) for kw in target_keywords]) + r')\b.*?has\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)(?!\s+[a-zA-Z])',
            # Pattern: "The [attribute] of [garment] is [value]"
            r'the\s+([a-zA-Z]+(?:\s+[a-zA-Z]+){0,2})\s+of\s+\b(' + '|'.join([re.escape(kw) for kw in target_keywords]) + r')\b.*?is\s+([a-zA-Z]+(?:\s+[a-zA-Z]+){0,2})',
            # Pattern: "[garment] with [attribute]" (1-2 words for attributes)
            r'\b(' + '|'.join([re.escape(kw) for kw in target_keywords]) + r')\b.*?with\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)(?!\s+[a-zA-Z])',
        ]

        # Try each pattern
        for pattern_idx, pattern in enumerate(anchoring_patterns):
            matches = re.finditer(pattern, sentence_lower)
            for match in matches:
                if len(match.groups()) >= 2:
                    # Extract captured groups
                    if pattern_idx == 2:  # Special pattern with 3 groups
                        attribute_value = match.group(1)  # attribute
                        garment_mention = match.group(2)  # garment
                        actual_value = match.group(3)    # value
                        # For this pattern, the actual value is what we want
                        attribute_value = actual_value
                    else:
                        garment_mention = match.group(1)
                        attribute_value = match.group(2)

                    # Determine attribute type by checking if it matches any known tags
                    attr_type = None
                    confidence = 0

                    # Check material tags
                    for mat_tag, mat_config in self.attributes['material'].items():
                        if any(trigger.lower() in attribute_value.lower() for trigger in mat_config.get('triggers', [])):
                            attr_type = 'material'
                            confidence = 90
                            break

                    # Check pattern tags (if not already found)
                    if not attr_type:
                        for pat_tag, pat_config in self.attributes['pattern'].items():
                            if any(trigger.lower() in attribute_value.lower() for trigger in pat_config.get('triggers', [])):
                                attr_type = 'pattern'
                                confidence = 85
                                break

                    # Check neckline tags
                    if not attr_type:
                        for neck_tag, neck_config in self.attributes['neckline'].items():
                            if any(trigger.lower() in attribute_value.lower() for trigger in neck_config.get('triggers', [])):
                                attr_type = 'neckline'
                                confidence = 80
                                break

                    # Check sleeve tags
                    if not attr_type:
                        for sleeve_tag, sleeve_config in self.attributes['sleeve'].items():
                            if any(trigger.lower() in attribute_value.lower() for trigger in sleeve_config.get('triggers', [])):
                                attr_type = 'sleeve'
                                confidence = 80
                                break

                    if attr_type:
                        # Validate that the captured value can be mapped to a canonical tag
                        canonical_tag = self._map_raw_value_to_tag(attribute_value, attr_type)
                        if canonical_tag:
                            anchored_attributes.append((garment_mention, attribute_value, attr_type, confidence))

        return anchored_attributes

    def _map_raw_value_to_tag(self, raw_value: str, attr_type: str) -> Optional[str]:
        """
        Map a raw attribute value to its canonical tag.

        Args:
            raw_value: Raw text like "cotton fabric"
            attr_type: Attribute type like "material"

        Returns:
            Canonical tag name or None if no match
        """
        raw_lower = raw_value.lower()

        for tag_name, tag_config in self.attributes[attr_type].items():
            for trigger in tag_config.get('triggers', []):
                if trigger.lower() in raw_lower:
                    return tag_name

        return None

    def _sentence_mentions_target(self, sentence: str, target_category: str) -> bool:
        """
        Check if a sentence mentions the target garment or its synonyms.
        """
        sentence_lower = sentence.lower()
        target_scope = self.get_garment_scope(target_category)
        target_keywords = self.garment_keywords.get(target_scope, [])

        # Check for exact category match
        if target_category.lower() in sentence_lower:
            return True

        # Check for synonym matches
        for keyword in target_keywords:
            if keyword.lower() in sentence_lower:
                return True

        return False

    def _select_primary_tag(
        self,
        tags: List[str],
        attr_name: str,
        category2: str,
        text: str
    ) -> str:
        """
        Select the best primary tag using intelligent prioritization.

        Priority order:
        1. Attributes from first-mentioned target garment sentences
        2. Anchored attributes
        3. Position-based (earlier in text)
        4. First tag (fallback)

        Args:
            tags: List of candidate tags
            attr_name: Attribute name (material, pattern, etc.)
            category2: Target category
            text: Full text for context

        Returns:
            Best primary tag or unknown
        """
        if not tags:
            return self.unknown_value

        if len(tags) == 1:
            return tags[0]

        sentences = self._split_into_sentences(text)

        # Priority 1: Tags from sentences mentioning the target garment (prioritize first-mentioned)
        if category2:
            target_sentences = []
            for i, sentence in enumerate(sentences):
                if self._sentence_mentions_target(sentence, category2):
                    target_sentences.append((i, sentence))

            # Sort by sentence index (earlier = higher priority)
            target_sentences.sort(key=lambda x: x[0])

            for sentence_idx, sentence in target_sentences:
                for tag in tags:
                    if tag.lower() in sentence.lower():
                        return tag

        # Priority 2: Anchored attributes
        anchored_found = []
        for sentence in sentences:
            if category2:
                anchored = self._anchor_attribute_to_subject(sentence, category2)
                for garment_mention, attr_value, attr_type, confidence in anchored:
                    if attr_type == attr_name and attr_value in tags:
                        anchored_found.append((attr_value, confidence))

        if anchored_found:
            anchored_found.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence
            return anchored_found[0][0]

        # Priority 3: Position-based (earlier in text)
        for sentence in sentences:
            for tag in tags:
                if tag.lower() in sentence.lower():
                    return tag

        # Fallback to first tag
        return tags[0]

    def _find_first_garment_mention(self, text: str, target_category: str) -> int:
        """
        Find the position of the first mention of the target garment in the text.

        Returns the character position, or -1 if not found.
        """
        target_keywords = self.garment_keywords.get(self.get_garment_scope(target_category), [])
        text_lower = text.lower()

        min_position = len(text)  # Start with end of text
        for keyword in target_keywords:
            pos = text_lower.find(keyword.lower())
            if pos != -1 and pos < min_position:
                min_position = pos

        return min_position if min_position < len(text) else -1

    def _get_garment_priority(
        self,
        sentence: str,
        target_category: str,
        sentence_index: int = 0,
        text_position: int = 0,
        full_text: str = ""
    ) -> int:
        """
        Calculate priority score for a sentence based on how closely it matches the target garment.

        Priority hierarchy (from highest to lowest):
        1. Sentences with anchored attributes for target garment (handled separately)
        2. Sentences mentioning exact target category
        3. Sentences mentioning target synonyms
        4. Sentences in target scope (upper/lower/full)
        5. First-mentioned garment in full text (position bonus)
        6. Outer clothing (lowest priority)

        Args:
            sentence: The sentence to evaluate
            target_category: The target category2
            sentence_index: Index of sentence in text (earlier = higher priority)
            text_position: Character position in full text
            full_text: Full text for first-mention detection

        Returns:
            Priority score (higher = better)
        """
        sentence_lower = sentence.lower()
        target_scope = self.get_garment_scope(target_category)
        base_score = 0

        # 1. Exact category match (highest priority)
        if target_category.lower() in sentence_lower:
            base_score = 100

        # 2. Synonym matches
        elif target_category.lower() in ['blouses', 'shirts', 't-shirts'] and 'shirt' in sentence_lower:
            base_score = 90
        elif target_category.lower() == 'sweatshirts' and 'sweater' in sentence_lower:
            base_score = 90
        elif target_category.lower() in ['jackets', 'coats'] and 'jacket' in sentence_lower:
            base_score = 90
        elif target_category.lower() in ['pants', 'jeans'] and 'pants' in sentence_lower:
            base_score = 90

        # 3. Scope matches
        else:
            sentence_scope = self._identify_sentence_scope(sentence)
            if sentence_scope:
                if sentence_scope == target_scope:
                    base_score = 80  # Same scope as target
                elif target_scope == 'full':
                    # For full-body garments, upper/lower clothing is relevant
                    if sentence_scope in ['upper', 'lower']:
                        base_score = 70
                    elif sentence_scope == 'full':
                        base_score = 85  # Explicit full-body mention

        # 4. First-mention bonus
        if full_text and base_score > 0:
            first_mention_pos = self._find_first_garment_mention(full_text, target_category)
            if first_mention_pos >= 0:
                # Sentences closer to first mention get bonus points
                distance_from_first = abs(text_position - first_mention_pos)
                proximity_bonus = max(0, 50 - distance_from_first // 10)  # Closer = higher bonus
                base_score += proximity_bonus

        # 5. Outer clothing (lowest priority)
        if 'outer clothing' in sentence_lower and base_score == 0:
            base_score = 10

        # 6. Sentence order bonus (earlier sentences slightly preferred)
        sentence_order_bonus = max(0, 10 - sentence_index)
        base_score += sentence_order_bonus

        return base_score
    
    def _get_compatible_attributes(self, scope: str) -> Set[str]:
        """
        Get list of compatible attributes for a garment scope.
        
        Args:
            scope: Garment scope ('upper', 'lower', 'full')
            
        Returns:
            Set of compatible attribute names
        """
        return set(self.attribute_compatibility.get(scope, ['material', 'pattern']))
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if not text:
            return []
        
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _identify_sentence_scope(self, sentence: str) -> Optional[str]:
        """
        Identify which garment scope a sentence is talking about.
        
        Args:
            sentence: A single sentence from the text
            
        Returns:
            Scope ('upper', 'lower', 'full') or None if no garment mentioned
        """
        sentence_lower = sentence.lower()
        
        # Check each scope's patterns
        for scope, patterns in self._garment_patterns.items():
            for pattern in patterns:
                if pattern.search(sentence_lower):
                    return scope
        
        return None
    
    def _filter_noise_sentence(self, sentence: str) -> bool:
        """
        Check if a sentence is noise (about accessories, etc.).
        
        Args:
            sentence: A single sentence
            
        Returns:
            True if the sentence is noise and should be skipped
        """
        sentence_lower = sentence.lower()
        
        for noise_phrase in self.noise_phrases:
            if noise_phrase.lower() in sentence_lower:
                return True
        
        return False
    
    def _extract_from_sentence(
        self, 
        sentence: str, 
        target_scope: str
    ) -> Dict[str, List[Tuple[str, int, int]]]:
        """
        Extract all attribute matches from a single sentence.
        
        Args:
            sentence: Sentence to extract from
            target_scope: The target garment scope
            
        Returns:
            Dict mapping attribute name to list of (tag, position, priority)
        """
        compatible_attrs = self._get_compatible_attributes(target_scope)
        results = {}
        
        for attr_name in self.attributes:
            if attr_name not in compatible_attrs:
                results[attr_name] = []
                continue
            
            matches = self._find_matches(sentence, attr_name)
            matches = self._apply_exclusions(sentence, attr_name, matches)
            results[attr_name] = matches
        
        return results
    
    def _find_matches(
        self, 
        text: str, 
        attr_name: str
    ) -> List[Tuple[str, int, int]]:
        """
        Find all matching tags for an attribute type.
        
        Args:
            text: Text to search
            attr_name: Attribute type (e.g., 'material')
            
        Returns:
            List of (tag_name, position, priority) tuples
        """
        matches = []
        text_lower = text.lower()
        
        if attr_name not in self._trigger_patterns:
            return matches
        
        for tag_name, patterns in self._trigger_patterns[attr_name].items():
            tag_config = self.attributes[attr_name][tag_name]
            priority = tag_config.get('priority', 50)
            
            for pattern in patterns:
                match = pattern.search(text_lower)
                if match:
                    matches.append((tag_name, match.start(), priority))
                    break  # Found a match for this tag, move to next tag
        
        return matches
    
    def _apply_exclusions(
        self, 
        text: str, 
        attr_name: str, 
        matches: List[Tuple[str, int, int]]
    ) -> List[Tuple[str, int, int]]:
        """
        Apply exclusion rules to filter out matches.
        
        Args:
            text: Original text
            attr_name: Attribute type
            matches: Current matches
            
        Returns:
            Filtered matches with exclusions applied
        """
        if not matches:
            return matches
        
        text_lower = text.lower()
        filtered = []
        
        # Get all matched tag names for cross-checking
        matched_tags = {m[0] for m in matches}
        
        for tag_name, position, priority in matches:
            tag_config = self.attributes[attr_name][tag_name]
            exclusions = tag_config.get('exclusions', [])
            
            # Check if any exclusion phrase is in the text
            should_exclude = False
            for exclusion in exclusions:
                exclusion_lower = exclusion.lower()
                
                # Check if exclusion is another matched tag
                if exclusion_lower in matched_tags:
                    should_exclude = True
                    break
                
                # Check if exclusion phrase is in text
                if exclusion_lower in text_lower:
                    should_exclude = True
                    break
            
            if not should_exclude:
                filtered.append((tag_name, position, priority))
        
        return filtered
    
    def _resolve_conflicts(
        self, 
        matches: List[Tuple[str, int, int]], 
        is_mutually_exclusive: bool
    ) -> List[str]:
        """
        Resolve conflicts when multiple tags match.
        
        Args:
            matches: List of (tag_name, position, priority) tuples
            is_mutually_exclusive: Whether only one tag should be returned
            
        Returns:
            List of resolved tag names
        """
        if not matches:
            return []
        
        if is_mutually_exclusive:
            # For mutually exclusive attributes, pick by:
            # 1. Lowest priority number (higher priority)
            # 2. First occurrence (position) as tiebreaker
            sorted_matches = sorted(matches, key=lambda x: (x[2], x[1]))
            return [sorted_matches[0][0]]
        else:
            # For multi-value attributes, sort by priority and return all
            sorted_matches = sorted(matches, key=lambda x: (x[2], x[1]))
            return [m[0] for m in sorted_matches]
    
    def _is_full_body_garment(self, category2: str) -> bool:
        """Check if category2 represents a full-body garment."""
        return self.get_garment_scope(category2) == 'full'

    def _handle_full_body_garment(
        self,
        text: str,
        category2: str,
        filter_noise: bool = True
    ) -> Dict[str, List[str]]:
        """
        Special handling for full-body garments (dresses, rompers, jumpsuits).

        These garments are often described as upper clothing (tank top, shirt) in text,
        but should extract upper attributes (neckline, sleeve) since they have them.
        """
        # For full-body garments, we extract upper attributes
        target_scope = 'upper'  # Extract upper attributes for full-body garments

        # Split into sentences
        sentences = self._split_into_sentences(text)

        # Collect matches with priority:
        # 1. Sentences explicitly about the full-body garment (dress, romper, etc.)
        # 2. Sentences about upper clothing (tank top, shirt, etc.)
        # 3. Other sentences as fallback

        primary_sentences = []  # About dress/romper specifically
        secondary_sentences = []  # About upper clothing
        tertiary_sentences = []  # Everything else

        for sentence in sentences:
            # Skip noise sentences
            if filter_noise and self._filter_noise_sentence(sentence):
                continue

            sentence_scope = self._identify_sentence_scope(sentence)

            if sentence_scope == 'full':
                primary_sentences.append(sentence)
            elif sentence_scope == 'upper':
                secondary_sentences.append(sentence)
            else:
                tertiary_sentences.append(sentence)

        # Extract from sentences in priority order
        all_sentences = primary_sentences + secondary_sentences + tertiary_sentences

        # Collect matches from all relevant sentences
        all_matches: Dict[str, List[Tuple[str, int, int]]] = {
            attr: [] for attr in self.attributes
        }

        for sentence_idx, sentence in enumerate(all_sentences):
            sentence_matches = self._extract_from_sentence(sentence, target_scope)
            for attr_name, matches in sentence_matches.items():
                # Add sentence index to each match (matches are (tag, position, priority))
                for tag, position, priority in matches:
                    all_matches[attr_name].append((tag, position, priority, sentence_idx))

        # Resolve conflicts and build final results
        results = {}
        for attr_name in self.attributes:
            is_exclusive = attr_name in self.mutually_exclusive
            tags = self._resolve_conflicts(all_matches[attr_name], is_exclusive)
            results[attr_name] = tags

        return results

    def extract(
        self,
        text: str,
        category2: Optional[str] = None,
        filter_noise: bool = True
    ) -> Dict[str, List[str]]:
        """
        Extract attributes from text for the target garment (category2).
        
        This method uses sentence-level parsing to:
        1. Identify which sentences refer to the target garment
        2. Only extract attributes from those sentences
        3. Filter out incompatible attributes (e.g., no sleeves for shorts)
        
        Args:
            text: Product description text
            category2: Product category (e.g., 'shorts', 'blouses')
            filter_noise: Whether to filter out noise sentences
            
        Returns:
            Dictionary mapping attribute names to lists of canonical tags
        """
        if not text:
            return {attr: [] for attr in self.attributes}

        # Special handling for full-body garments
        if category2 and self._is_full_body_garment(category2):
            return self._handle_full_body_garment(text, category2, filter_noise)

        # Determine target scope from category2
        has_explicit_category = bool(category2 and category2.strip())
        target_scope = self.get_garment_scope(category2) if has_explicit_category else 'upper'
        compatible_attrs = self._get_compatible_attributes(target_scope) if has_explicit_category else set(self.attributes.keys())
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        # Collect matches from sentences about the target garment
        all_matches: Dict[str, List[Tuple[str, int, int]]] = {
            attr: [] for attr in self.attributes
        }
        
        # Collect sentences with their priorities and original indices
        sentence_info = []
        text = ' '.join(sentences)  # Reconstruct full text for first-mention detection
        current_pos = 0

        for i, sentence in enumerate(sentences):
            # Skip noise sentences
            if filter_noise and self._filter_noise_sentence(sentence):
                continue

            priority = self._get_garment_priority(
                sentence, category2,
                sentence_index=i,
                text_position=current_pos,
                full_text=text
            ) if has_explicit_category else 50
            sentence_info.append((sentence, priority, i))  # (sentence, priority, original_index)
            current_pos += len(sentence) + 1  # +1 for space

        # Sort sentences by priority (highest first)
        sentence_info.sort(key=lambda x: x[1], reverse=True)

        # Extract from sentences in priority order
        # For each attribute, we'll collect matches but prioritize higher-priority sentences
        for sentence_idx, (sentence, priority, original_idx) in enumerate(sentence_info):
            # Use detected scope if no explicit category, otherwise use target scope
            sentence_scope = self._identify_sentence_scope(sentence) if not has_explicit_category else target_scope
            extraction_scope = sentence_scope or target_scope

            # First, try to get anchored attributes (higher priority)
            if has_explicit_category:
                anchored = self._anchor_attribute_to_subject(sentence, category2)
                for garment_mention, attr_value, attr_type, confidence in anchored:
                    # Only use anchored attributes that match our target category
                    if attr_type in compatible_attrs:
                        # Map the raw attr_value to a canonical tag
                        canonical_tag = self._map_raw_value_to_tag(attr_value, attr_type)
                        if canonical_tag:
                            # Convert confidence to priority score (higher confidence = higher priority)
                            all_matches[attr_type].append((canonical_tag, 0, confidence, original_idx))

            # Extract attributes from this sentence (regular extraction)
            sentence_matches = self._extract_from_sentence(sentence, extraction_scope)

            for attr_name, matches in sentence_matches.items():
                # Only add matches if this attribute is compatible
                if has_explicit_category and attr_name not in compatible_attrs:
                    continue
                all_matches[attr_name].extend(matches)
        
        # Filter tag contamination for explicit categories
        filtered_matches = all_matches.copy()

        if has_explicit_category:
            # Track which sentences have anchored attributes for target garment
            anchored_sentences = set()
            for sentence, priority, original_idx in sentence_info:
                if self._anchor_attribute_to_subject(sentence, category2):
                    anchored_sentences.add(sentence)

            # Filter matches to only include those relevant to target garment
            for attr_name in self.attributes:
                filtered_list = []
                for match in all_matches[attr_name]:
                    if len(match) == 4:
                        tag, position, priority, sentence_idx = match
                    elif len(match) == 3:
                        tag, position, priority = match
                        sentence_idx = 0  # fallback
                    else:
                        continue
                    sentence = sentences[sentence_idx] if sentence_idx < len(sentences) else ""

                    # For explicit categories, be strict: only include attributes that are clearly about the target garment
                    should_include = False

                    # Include if priority > 80 (anchored attributes have high priority)
                    if priority > 80:
                        should_include = True
                    # Include if sentence mentions target garment
                    elif sentence and self._sentence_mentions_target(sentence, category2):
                        should_include = True
                    # Include if from anchored sentence (contains anchored attributes for target)
                    elif sentence in anchored_sentences:
                        should_include = True

                    if should_include:
                        if len(match) == 4:
                            filtered_list.append(match)  # Keep original format
                        else:
                            filtered_list.append((tag, position, priority, sentence_idx))

                filtered_matches[attr_name] = filtered_list

        # Resolve conflicts and build final results
        results = {}
        for attr_name in self.attributes:
            is_exclusive = attr_name in self.mutually_exclusive

            # Filter to only compatible attributes (when category2 is provided)
            if has_explicit_category and attr_name not in compatible_attrs:
                results[attr_name] = []
            else:
                # Extract just (tag, position, priority) for conflict resolution
                simplified_matches = []
                for match in filtered_matches[attr_name]:
                    if len(match) >= 3:
                        tag, pos, pri = match[0], match[1], match[2]
                        simplified_matches.append((tag, pos, pri))
                tags = self._resolve_conflicts(simplified_matches, is_exclusive)
                results[attr_name] = tags
        
        return results
    
    def extract_with_primary(
        self, 
        text: str, 
        category2: Optional[str] = None,
        filter_noise: bool = True
    ) -> Dict[str, Any]:
        """
        Extract attributes with both full list and primary tag.
        
        Args:
            text: Product description text
            category2: Product category (optional)
            filter_noise: Whether to filter out noise sentences
            
        Returns:
            Dictionary with 'attr_{name}' (list) and 'attr_{name}_primary' (str) keys
        """
        raw_results = self.extract(text, category2, filter_noise)

        results = {}
        for attr_name, tags in raw_results.items():
            results[f'attr_{attr_name}'] = tags
            # Use smart primary selection
            results[f'attr_{attr_name}_primary'] = self._select_primary_tag(tags, attr_name, category2 or '', text)
        
        return results
    
    def process_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single dataset example to add attribute columns.
        
        This method is designed to be used with dataset.map().
        
        Args:
            example: Dataset example with 'text' and optionally 'category2' fields
            
        Returns:
            Example with added attribute fields
        """
        text = example.get('text', '')
        category2 = example.get('category2', '')
        
        extracted = self.extract_with_primary(text, category2)
        
        # Merge extracted attributes into example
        example.update(extracted)
        
        return example
    
    def get_attribute_names(self) -> List[str]:
        """Get list of attribute type names."""
        return list(self.attributes.keys())
    
    def get_canonical_tags(self, attr_name: str) -> List[str]:
        """Get list of canonical tags for an attribute type."""
        if attr_name not in self.attributes:
            return []
        return list(self.attributes[attr_name].keys())
    
    def get_all_triggers(self, attr_name: str, tag_name: str) -> List[str]:
        """Get all trigger phrases for a specific tag."""
        if attr_name not in self.attributes:
            return []
        if tag_name not in self.attributes[attr_name]:
            return []
        return self.attributes[attr_name][tag_name].get('triggers', [])


def create_extractor(schema_path: Optional[str] = None) -> AttributeExtractor:
    """
    Factory function to create an AttributeExtractor instance.
    
    Args:
        schema_path: Path to schema YAML file (optional)
        
    Returns:
        Configured AttributeExtractor instance
    """
    return AttributeExtractor(schema_path)


# Convenience function for dataset processing
def extract_attributes_batch(
    examples: Dict[str, List],
    extractor: Optional[AttributeExtractor] = None
) -> Dict[str, List]:
    """
    Batch extraction function for use with dataset.map(batched=True).
    
    Args:
        examples: Batch of examples (dict of lists)
        extractor: AttributeExtractor instance (creates one if None)
        
    Returns:
        Batch with added attribute columns
    """
    if extractor is None:
        extractor = AttributeExtractor()
    
    texts = examples.get('text', [])
    category2s = examples.get('category2', [''] * len(texts))
    
    # Initialize result columns
    attr_names = extractor.get_attribute_names()
    for attr_name in attr_names:
        examples[f'attr_{attr_name}'] = []
        examples[f'attr_{attr_name}_primary'] = []
    
    # Process each example
    for text, cat2 in zip(texts, category2s):
        extracted = extractor.extract_with_primary(text, cat2)
        for key, value in extracted.items():
            examples[key].append(value)
    
    return examples
