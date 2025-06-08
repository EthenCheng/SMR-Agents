
import re
from tqdm import tqdm
from utils.output_utils import format_json_out_put, filter_finished, ensure_dir, format_output_filepath
from utils.register import register_class, registry
from methods.base_method import BaseMethod
from methods.scene_graph_refiner import SceneGraphRefiner
from prompts.prompts import (
    get_description_prompt,
    get_expert_consultation_prompt,
    get_expert_opinions_prompt,
    get_expert_diagnosis_prompt,
    get_expert_evaluation_prompt,
    get_expert_evaluation_followup_prompt,
    get_specialists_rethink_prompt,
    get_diagnostic_reassessment_prompt,
    get_integration_summary_prompt
)


@register_class(alias="SMRAgents")
class SMRAgents(BaseMethod):
    
    def __init__(self, dataset, args):
        """
        Initialize SMRAgents method
        
        Args:
            dataset: Medical QA dataset
            args: Configuration arguments
        """
        self.dataset = dataset
        self.output_file_path = format_output_filepath(
            args.language_model_name, 
            args.visual_model_name, 
            args.method, 
            args.dataset_name
        )
        ensure_dir(self.output_file_path)
        
        self.max_retries = args.max_retries
        self.v_engine = registry.get_class(args.visual_model_name)(device=args.v_device)
        self.l_engine = registry.get_class(args.language_model_name)(device=args.l_device)
        self.ff_print = args.ff_print
        
        # Initialize scene graph refiner if knowledge base is available
        self.scene_graph_refiner = None
        if hasattr(args, 'knowledge_base_path') and args.knowledge_base_path:
            self.scene_graph_refiner = SceneGraphRefiner(
                args.knowledge_base_path,
                self.l_engine
            )
            print(f"Initialized scene graph refiner with knowledge base at {args.knowledge_base_path}")
    
    def generate_and_refine_scene_graph(self, question: str, img, img_path: str) -> str:
        """
        Generate initial scene graph and refine it using knowledge base
        
        Args:
            question: Medical question
            img: Image data
            img_path: Path to image
            
        Returns:
            Refined scene graph text
        """
        # Step 1: Generate initial scene graph
        initial_scene_graph = self.v_engine.get_response(
            get_description_prompt(question), 
            img, 
            img_path
        )
        
        if self.ff_print:
            print(f"Initial Scene Graph: {initial_scene_graph[:500]}...")
        
        # Step 2: Refine scene graph using knowledge base
        if self.scene_graph_refiner:
            refined_scene_graph = self.scene_graph_refiner.refine_scene_graph(
                initial_scene_graph,
                verbose=self.ff_print
            )
            
            # Validate refined scene graph
            validation_results = self.scene_graph_refiner.validate_scene_graph(refined_scene_graph)
            if not validation_results["is_valid"]:
                print(f"Warning: Refined scene graph validation failed: {validation_results['errors']}")
                # Fall back to initial scene graph if refinement fails
                return initial_scene_graph
            
            if self.ff_print:
                print(f"Refined Scene Graph: {refined_scene_graph[:500]}...")
                if validation_results["warnings"]:
                    print(f"Validation warnings: {validation_results['warnings']}")
            
            return refined_scene_graph
        else:
            # No knowledge base available, use initial scene graph
            return initial_scene_graph
    
    def extract_specialists_from_consultation(self, consultation_text: str) -> list:
        """
        Extract specialist information from consultation response
        
        Args:
            consultation_text: Expert consultation response text
            
        Returns:
            List of specialists mentioned
        """
        # Extract specialists using pattern matching
        specialists = []
        pattern = r'Expert\s*:\s*(.*?)(?=Expert\s*:|$)'
        matches = re.findall(pattern, consultation_text, re.DOTALL)
        
        for match in matches:
            specialist_info = match.strip()
            if specialist_info:
                specialists.append(specialist_info)
        
        return specialists
    
    def process_feedback_iteration(self, question: str, description: str, 
                                 experts_opinions: str, expert_diagnosis: str,
                                 iteration: int) -> tuple:
        """
        Process one iteration of feedback and refinement
        
        Args:
            question: Medical question
            description: Scene graph description
            experts_opinions: Current expert opinions
            expert_diagnosis: Current diagnostic expert's diagnosis
            iteration: Current iteration number
            
        Returns:
            Updated (experts_opinions, expert_diagnosis, continue_iteration)
        """
        if iteration == 0:
            # First iteration - full evaluation
            expert_evaluation = self.l_engine.get_response(
                get_expert_evaluation_prompt(question, description, expert_diagnosis, experts_opinions)
            )
        else:
            # Follow-up evaluation
            expert_evaluation = self.l_engine.get_response(
                get_expert_evaluation_followup_prompt(question, description, experts_opinions, expert_diagnosis)
            )
        
        if self.ff_print:
            print(f"Iteration {iteration + 1} - Review Expert's Evaluation: {expert_evaluation[:300]}...")
        
        # Check if all opinions are consistent
        if "all opinions are consistent" in expert_evaluation.lower():
            return experts_opinions, expert_diagnosis, False
        
        # Process specialist feedback
        if "Feedback to Specialist Experts:" in expert_evaluation:
            feedback_start = expert_evaluation.find("Feedback to Specialist Experts:")
            feedback_end = expert_evaluation.find("Feedback to Diagnostic Specialist:")
            if feedback_end == -1:
                feedback_end = len(expert_evaluation)
            
            feedback_to_specialists = expert_evaluation[feedback_start:feedback_end].strip()
            
            if feedback_to_specialists:
                updated_opinions = self.update_specialist_opinions(
                    question, description, experts_opinions, feedback_to_specialists
                )
                experts_opinions = updated_opinions
        
        # Process diagnostic specialist feedback
        if "Feedback to Diagnostic Specialist:" in expert_evaluation:
            expert_diagnosis = self.l_engine.get_response(
                get_diagnostic_reassessment_prompt(question, description, experts_opinions)
            )
            if self.ff_print:
                print(f"Updated Diagnostic Expert's Diagnosis: {expert_diagnosis[:300]}...")
        
        return experts_opinions, expert_diagnosis, True
    
    def update_specialist_opinions(self, question: str, description: str, 
                                 current_opinions: str, feedback: str) -> str:
        """
        Update specialist opinions based on feedback
        
        Args:
            question: Medical question
            description: Scene graph description
            current_opinions: Current specialist opinions
            feedback: Feedback from review expert
            
        Returns:
            Updated opinions text
        """
        updated_opinions_list = []
        
        # Extract specialists from current opinions
        specialists = re.findall(r'Expert\s*.*?:\s*(.*?)(?=Expert\s*|$)', current_opinions, re.DOTALL)
        
        for specialist_text in specialists:
            # Check if this specialist has feedback
            specialist_name = specialist_text.split('\n')[0].strip()
            
            # Look for feedback specific to this specialist
            if specialist_name in feedback:
                # Get updated opinion from specialist
                specialist_feedback_prompt = get_specialists_rethink_prompt(
                    question, description, feedback
                )
                updated_opinion = self.l_engine.get_response(specialist_feedback_prompt)
                
                if self.ff_print:
                    print(f"Updated opinion for {specialist_name}: {updated_opinion[:200]}...")
                
                updated_opinions_list.append(f"Expert: {specialist_name}\n{updated_opinion}")
            else:
                # Keep original opinion
                updated_opinions_list.append(f"Expert: {specialist_text}")
        
        return "\n\n".join(updated_opinions_list)
    
    def run(self):
        for round_count in range(self.max_retries):
            print(f"Start {round_count + 1} round of answering questions.")
            todo_list = filter_finished(len(self.dataset), self.output_file_path)
            
            if not todo_list:
                print("All questions have been answered.")
                return
            
            for idx in tqdm(todo_list, desc="Processing questions"):
                try:
                    img, question, answer, img_path = self.dataset[idx]
                    
                    # Step 1: Generate and refine scene graph
                    description = self.generate_and_refine_scene_graph(question, img, img_path)
                    
                    # Step 2: Expert Consultation
                    expert_consultation = self.l_engine.get_response(
                        get_expert_consultation_prompt(question, description)
                    )
                    if self.ff_print:
                        print(f"Expert Consultation: {expert_consultation[:300]}...")
                    
                    # Step 3: Specialists provide reasoning and answers
                    experts_opinions = self.l_engine.get_response(
                        get_expert_opinions_prompt(question, description, expert_consultation)
                    )
                    if self.ff_print:
                        print(f"Specialists' Opinions: {experts_opinions[:300]}...")
                    
                    # Step 4: Diagnostic Expert provides diagnosis
                    expert_diagnosis = self.l_engine.get_response(
                        get_expert_diagnosis_prompt(question, description, experts_opinions)
                    )
                    if self.ff_print:
                        print(f"Diagnostic Expert's Diagnosis: {expert_diagnosis[:300]}...")
                    
                    # Step 5: Iterative feedback and refinement
                    max_iterations = 3
                    for iteration in range(max_iterations):
                        experts_opinions, expert_diagnosis, should_continue = self.process_feedback_iteration(
                            question, description, experts_opinions, expert_diagnosis, iteration
                        )
                        
                        if not should_continue:
                            break
                    
                    # Step 6: Integration Summary (Final Answer)
                    all_expert_opinions = f"{experts_opinions}\n\n{expert_diagnosis}"
                    final_answer = self.l_engine.get_response(
                        get_integration_summary_prompt(question, description, all_expert_opinions)
                    )
                    if self.ff_print:
                        print(f"Final Answer: {final_answer[:300]}...")
                    
                    # Format and save final answer
                    format_json_out_put(question, answer, final_answer, idx, self.output_file_path)
                    
                except Exception as e:
                    print(f"Error processing question {idx}: {str(e)}")
                    # Save error result
                    format_json_out_put(
                        question, 
                        answer, 
                        f"Error: {str(e)}", 
                        idx, 
                        self.output_file_path
                    )