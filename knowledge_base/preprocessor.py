"""
Knowledge Base Preprocessor Module
Preprocesses RadGraph and TCGA-Reports datasets into triplet format
"""

import json
import os
import pickle
from typing import List, Tuple, Dict
import pandas as pd
from tqdm import tqdm


class KnowledgeBasePreprocessor:
    """Preprocesses medical knowledge bases into triplet format for efficient retrieval"""
    
    def __init__(self, radgraph_path: str, tcga_reports_path: str, output_dir: str):
        """
        Initialize the preprocessor
        
        Args:
            radgraph_path: Path to RadGraph dataset
            tcga_reports_path: Path to TCGA-Reports dataset
            output_dir: Directory to save processed triplets
        """
        self.radgraph_path = radgraph_path
        self.tcga_reports_path = tcga_reports_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def process_radgraph(self) -> List[Tuple[str, str, str]]:
        """
        Process RadGraph dataset into triplets
        
        Returns:
            List of triplets (subject, predicate, object)
        """
        triplets = []
        
        # Load RadGraph data
        with open(self.radgraph_path, 'r', encoding='utf-8') as f:
            radgraph_data = json.load(f)
        
        print("Processing RadGraph dataset...")
        for report_id, report_data in tqdm(radgraph_data.items()):
            if 'entities' in report_data and 'relations' in report_data:
                entities = report_data['entities']
                relations = report_data['relations']
                
                # Extract entity information
                entity_map = {}
                for entity_id, entity_info in entities.items():
                    entity_map[entity_id] = {
                        'label': entity_info.get('label', ''),
                        'type': entity_info.get('label_type', ''),
                        'attributes': entity_info.get('attributes', {})
                    }
                
                # Create triplets from relations
                for relation in relations:
                    if isinstance(relation, dict):
                        subject_id = relation.get('subject', '')
                        object_id = relation.get('object', '')
                        predicate = relation.get('type', '')
                        
                        if subject_id in entity_map and object_id in entity_map:
                            subject = f"{entity_map[subject_id]['label']}:{entity_map[subject_id]['type']}"
                            obj = f"{entity_map[object_id]['label']}:{entity_map[object_id]['type']}"
                            triplets.append((subject, predicate, obj))
                
                # Create attribute triplets
                for entity_id, entity_info in entity_map.items():
                    entity_label = f"{entity_info['label']}:{entity_info['type']}"
                    for attr_key, attr_value in entity_info['attributes'].items():
                        triplets.append((entity_label, f"has_{attr_key}", str(attr_value)))
        
        return triplets
    
    def process_tcga_reports(self) -> List[Tuple[str, str, str]]:
        """
        Process TCGA-Reports dataset into triplets
        
        Returns:
            List of triplets (subject, predicate, object)
        """
        triplets = []
        
        # Load TCGA-Reports data
        if self.tcga_reports_path.endswith('.json'):
            with open(self.tcga_reports_path, 'r', encoding='utf-8') as f:
                tcga_data = json.load(f)
        elif self.tcga_reports_path.endswith('.csv'):
            tcga_data = pd.read_csv(self.tcga_reports_path).to_dict('records')
        else:
            raise ValueError("Unsupported file format for TCGA-Reports")
        
        print("Processing TCGA-Reports dataset...")
        for report in tqdm(tcga_data):
            # Extract structured information from reports
            # This is a simplified version - adjust based on actual TCGA format
            if 'findings' in report:
                findings = report['findings']
                if isinstance(findings, dict):
                    for organ, finding_list in findings.items():
                        for finding in finding_list:
                            if isinstance(finding, dict):
                                entity = finding.get('entity', '')
                                attributes = finding.get('attributes', {})
                                
                                # Create entity-organ relationship
                                if entity and organ:
                                    triplets.append((entity, 'located_in', organ))
                                
                                # Create attribute triplets
                                for attr_key, attr_value in attributes.items():
                                    if entity:
                                        triplets.append((entity, f"has_{attr_key}", str(attr_value)))
        
        return triplets
    
    def create_entity_index(self, triplets: List[Tuple[str, str, str]]) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Create an index for efficient entity-based retrieval
        
        Args:
            triplets: List of triplets
            
        Returns:
            Dictionary mapping entities to their related triplets
        """
        entity_index = {}
        
        for triplet in triplets:
            subject, predicate, obj = triplet
            
            # Index by subject
            if subject not in entity_index:
                entity_index[subject] = []
            entity_index[subject].append(triplet)
            
            # Index by object (for bidirectional retrieval)
            if obj not in entity_index:
                entity_index[obj] = []
            entity_index[obj].append(triplet)
        
        return entity_index
    
    def save_processed_data(self, radgraph_triplets: List[Tuple[str, str, str]], 
                          tcga_triplets: List[Tuple[str, str, str]]):
        """
        Save processed triplets and indices
        
        Args:
            radgraph_triplets: Triplets from RadGraph
            tcga_triplets: Triplets from TCGA-Reports
        """
        # Combine all triplets
        all_triplets = radgraph_triplets + tcga_triplets
        
        # Create entity index
        entity_index = self.create_entity_index(all_triplets)
        
        # Save triplets
        triplets_path = os.path.join(self.output_dir, 'medical_triplets.pkl')
        with open(triplets_path, 'wb') as f:
            pickle.dump(all_triplets, f)
        
        # Save entity index
        index_path = os.path.join(self.output_dir, 'entity_index.pkl')
        with open(index_path, 'wb') as f:
            pickle.dump(entity_index, f)
        
        # Save metadata
        metadata = {
            'total_triplets': len(all_triplets),
            'radgraph_triplets': len(radgraph_triplets),
            'tcga_triplets': len(tcga_triplets),
            'unique_entities': len(entity_index)
        }
        
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(all_triplets)} triplets and entity index to {self.output_dir}")
        print(f"Metadata: {metadata}")
    
    def preprocess(self):
        """Run the complete preprocessing pipeline"""
        # Process RadGraph
        radgraph_triplets = self.process_radgraph()
        print(f"Extracted {len(radgraph_triplets)} triplets from RadGraph")
        
        # Process TCGA-Reports
        tcga_triplets = self.process_tcga_reports()
        print(f"Extracted {len(tcga_triplets)} triplets from TCGA-Reports")
        
        # Save processed data
        self.save_processed_data(radgraph_triplets, tcga_triplets)


if __name__ == "__main__":
    preprocessor = KnowledgeBasePreprocessor(
        radgraph_path="knowledge_base/data/radgraph/radgraph_data.json",
        tcga_reports_path="knowledge_base/data/tcga_reports/tcga_reports.json",
        output_dir="knowledge_base/processed"
    )
    preprocessor.preprocess()