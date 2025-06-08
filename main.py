import argparse
import os
import sys
from utils.register import registry
from scripts.SMRAgents import SMRAgents
from knowledge_base.preprocessor import KnowledgeBasePreprocessor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Medical Scene Graph Construction with Knowledge Base')
    
    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Name of the medical QA dataset')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the dataset')
    
    # Model arguments
    parser.add_argument('--language_model_name', type=str, default='gpt-3.5-turbo',
                       help='Language model to use')
    parser.add_argument('--visual_model_name', type=str, default='qwen2-VL-7B',
                       help='Visual model to use')
    parser.add_argument('--l_device', type=str, default='cuda',
                       help='Device for language model')
    parser.add_argument('--v_device', type=str, default='cuda',
                       help='Device for visual model')
    
    # Method arguments
    parser.add_argument('--method', type=str, default='SMRAgents',
                       help='Method to use')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='Maximum number of retries')
    parser.add_argument('--ff_print', action='store_true',
                       help='Enable verbose printing')
    
    # Knowledge base arguments
    parser.add_argument('--knowledge_base_path', type=str, default='knowledge_base/processed',
                       help='Path to processed knowledge base')
    parser.add_argument('--preprocess_kb', action='store_true',
                       help='Preprocess knowledge base before running')
    parser.add_argument('--radgraph_path', type=str,
                       help='Path to RadGraph dataset (for preprocessing)')
    parser.add_argument('--tcga_reports_path', type=str,
                       help='Path to TCGA-Reports dataset (for preprocessing)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory to save outputs')
    
    return parser.parse_args()


def preprocess_knowledge_base(args):
    """Preprocess knowledge base if requested"""
    if args.preprocess_kb:
        if not args.radgraph_path or not args.tcga_reports_path:
            raise ValueError("Both --radgraph_path and --tcga_reports_path must be provided for preprocessing")
        
        print("Preprocessing knowledge base...")
        preprocessor = KnowledgeBasePreprocessor(
            radgraph_path=args.radgraph_path,
            tcga_reports_path=args.tcga_reports_path,
            output_dir=args.knowledge_base_path
        )
        preprocessor.preprocess()
        print("Knowledge base preprocessing completed!")


def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Preprocess knowledge base if requested
    preprocess_knowledge_base(args)
    
    # Check if knowledge base exists
    if not os.path.exists(args.knowledge_base_path):
        print(f"Warning: Knowledge base not found at {args.knowledge_base_path}")
        print("The system will run without knowledge base enhancement.")
        response = input("Continue without knowledge base? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_dataset(args)
    print(f"Loaded {len(dataset)} examples")
    
    # Initialize and run SMRAgents
    print(f"Initializing SMRAgents method...")
    smragents = SMRAgents(dataset, args)
    
    print("Starting SMRAgents processing...")
    smragents.run()
    
    print("Processing completed!")


if __name__ == "__main__":
    main()