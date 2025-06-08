"""
Knowledge Base Retriever Module
Retrieves relevant medical knowledge from preprocessed triplets
"""

import json
import pickle
import os
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeBaseRetriever:
    """Retrieves relevant medical knowledge from preprocessed knowledge bases"""
    
    def __init__(self, processed_data_dir: str):
        """
        Initialize the retriever
        
        Args:
            processed_data_dir: Directory containing processed triplets and indices
        """
        self.processed_data_dir = processed_data_dir
        self.triplets = []
        self.entity_index = {}
        self.tfidf_vectorizer = None
        self.entity_embeddings = None
        self.entity_list = []
        
        self._load_processed_data()
        self._build_semantic_index()
    
    def _load_processed_data(self):
        """Load preprocessed triplets and indices"""
        # Load triplets
        triplets_path = os.path.join(self.processed_data_dir, 'medical_triplets.pkl')
        with open(triplets_path, 'rb') as f:
            self.triplets = pickle.load(f)
        
        # Load entity index
        index_path = os.path.join(self.processed_data_dir, 'entity_index.pkl')
        with open(index_path, 'rb') as f:
            self.entity_index = pickle.load(f)
        
        print(f"Loaded {len(self.triplets)} triplets and {len(self.entity_index)} unique entities")
    
    def _build_semantic_index(self):
        """Build semantic index for similarity-based retrieval"""
        # Extract all unique entities
        self.entity_list = list(self.entity_index.keys())
        
        # Create TF-IDF vectorizer for entity matching
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            lowercase=True,
            max_features=5000
        )
        
        # Fit and transform entity names
        entity_texts = [entity.lower().replace(':', ' ').replace('_', ' ') for entity in self.entity_list]
        self.entity_embeddings = self.tfidf_vectorizer.fit_transform(entity_texts)
    
    def extract_entities_from_scene_graph(self, scene_graph: Dict) -> List[str]:
        """
        Extract entities from a medical scene graph
        
        Args:
            scene_graph: Medical scene graph in JSON format
            
        Returns:
            List of entity labels
        """
        entities = []
        
        # Extract from objects
        if 'objects' in scene_graph:
            for obj in scene_graph['objects']:
                if 'type' in obj:
                    entities.append(obj['type'])
                if 'attributes' in obj:
                    for attr_key, attr_value in obj['attributes'].items():
                        entities.append(f"{obj.get('type', 'entity')}:{attr_key}:{attr_value}")
        
        # Extract from conditions
        if 'conditions' in scene_graph:
            for condition in scene_graph['conditions']:
                if 'type' in condition:
                    entities.append(condition['type'])
                if 'description' in condition:
                    entities.append(condition['description'])
        
        return entities
    
    def find_similar_entities(self, query_entity: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar entities using semantic similarity
        
        Args:
            query_entity: Entity to search for
            top_k: Number of similar entities to return
            
        Returns:
            List of (entity, similarity_score) tuples
        """
        # Transform query entity
        query_text = query_entity.lower().replace(':', ' ').replace('_', ' ')
        query_vector = self.tfidf_vectorizer.transform([query_text])
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, self.entity_embeddings).flatten()
        
        # Get top-k similar entities
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        similar_entities = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Similarity threshold
                similar_entities.append((self.entity_list[idx], similarities[idx]))
        
        return similar_entities
    
    def retrieve_entity_knowledge(self, entity: str, max_triplets: int = 10) -> List[Tuple[str, str, str]]:
        """
        Retrieve knowledge triplets related to an entity
        
        Args:
            entity: Entity to retrieve knowledge for
            max_triplets: Maximum number of triplets to return
            
        Returns:
            List of relevant triplets
        """
        retrieved_triplets = []
        
        # Direct lookup
        if entity in self.entity_index:
            retrieved_triplets.extend(self.entity_index[entity][:max_triplets//2])
        
        # Find similar entities
        similar_entities = self.find_similar_entities(entity, top_k=3)
        for similar_entity, _ in similar_entities:
            if similar_entity in self.entity_index:
                remaining_slots = max_triplets - len(retrieved_triplets)
                if remaining_slots > 0:
                    retrieved_triplets.extend(
                        self.entity_index[similar_entity][:remaining_slots//len(similar_entities)]
                    )
        
        return retrieved_triplets[:max_triplets]
    
    def retrieve_relationship_knowledge(self, subject: str, predicate: str, obj: str) -> List[Tuple[str, str, str]]:
        """
        Retrieve knowledge about specific relationships
        
        Args:
            subject: Subject entity
            predicate: Relationship type
            obj: Object entity
            
        Returns:
            List of relevant triplets
        """
        retrieved_triplets = []
        
        # Look for similar relationships
        for triplet in self.triplets:
            t_subject, t_predicate, t_object = triplet
            
            # Check for similar predicates
            if predicate.lower() in t_predicate.lower() or t_predicate.lower() in predicate.lower():
                # Check if entities are related
                if (subject.lower() in t_subject.lower() or t_subject.lower() in subject.lower() or
                    obj.lower() in t_object.lower() or t_object.lower() in obj.lower()):
                    retrieved_triplets.append(triplet)
        
        return retrieved_triplets[:5]
    
    def retrieve_knowledge_for_scene_graph(self, scene_graph: Dict, max_knowledge_per_entity: int = 5) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Retrieve relevant knowledge for an entire scene graph
        
        Args:
            scene_graph: Medical scene graph in JSON format
            max_knowledge_per_entity: Maximum triplets per entity
            
        Returns:
            Dictionary mapping entities to their relevant knowledge triplets
        """
        knowledge_dict = defaultdict(list)
        
        # Extract entities from scene graph
        entities = self.extract_entities_from_scene_graph(scene_graph)
        
        # Retrieve knowledge for each entity
        for entity in entities:
            entity_knowledge = self.retrieve_entity_knowledge(entity, max_knowledge_per_entity)
            if entity_knowledge:
                knowledge_dict[entity] = entity_knowledge
        
        # Retrieve knowledge for relationships
        if 'relationships' in scene_graph:
            for relation in scene_graph['relationships']:
                subject_id = relation.get('subject', '')
                predicate = relation.get('predicate', '')
                object_id = relation.get('object', '')
                
                # Find actual entity names from IDs
                subject_entity = None
                object_entity = None
                
                for obj in scene_graph.get('objects', []):
                    if obj.get('id') == subject_id:
                        subject_entity = obj.get('type', '')
                    if obj.get('id') == object_id:
                        object_entity = obj.get('type', '')
                
                if subject_entity and object_entity:
                    relation_knowledge = self.retrieve_relationship_knowledge(
                        subject_entity, predicate, object_entity
                    )
                    if relation_knowledge:
                        relation_key = f"{subject_entity}-{predicate}-{object_entity}"
                        knowledge_dict[relation_key] = relation_knowledge
        
        return dict(knowledge_dict)
    
    def format_retrieved_knowledge(self, knowledge_dict: Dict[str, List[Tuple[str, str, str]]]) -> str:
        """
        Format retrieved knowledge into a readable string
        
        Args:
            knowledge_dict: Dictionary of retrieved knowledge
            
        Returns:
            Formatted string of knowledge
        """
        formatted_knowledge = []
        
        for entity, triplets in knowledge_dict.items():
            formatted_knowledge.append(f"\n**Knowledge about '{entity}':**")
            for triplet in triplets:
                subject, predicate, obj = triplet
                formatted_knowledge.append(f"- {subject} {predicate} {obj}")
        
        return "\n".join(formatted_knowledge)


if __name__ == "__main__":
    # Retrieve knowledge
    knowledge = retriever.retrieve_knowledge_for_scene_graph(scene_graph)
    formatted = retriever.format_retrieved_knowledge(knowledge)
    print(formatted)