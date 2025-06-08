
def get_description_prompt(question):
    """Generate initial medical scene graph generation prompt"""
    prompt = (
        f"As a medical assistant specialized in generating detailed Medical Scene Graphs based on questions and images,\n"
        f"your task is to carefully analyze the provided image and generate a Medical Scene Graph in JSON format that is highly relevant to the following question: \"{question}\". Your analysis should focus on:\n\n"
    
        "1. **Identifying Medical Entities**:\n"
        "- Focus on identifying all medical entities in the image that are directly pertinent to answering the question. Each entity should be represented as an object with:\n"
        "  - A unique `id` for identification.\n"
        "  - A `type` specifying the nature of the entity (e.g., anatomical structure, lesion, medical device, imaging modality).\n"
        "  - A set of `attributes` describing diagnostically relevant features such as size, shape, function, appearance, location, intensity, texture, color, and other diagnostic characteristics.\n\n"
    
        "2. **Specifying Entity Relationships**:\n"
        "- Define meaningful relationships between objects that help understand the context and answer the question. Each relationship should include:\n"
        "  - `subject`: The id of the first object involved in the relationship.\n"
        "  - `predicate`: The type of relationship (e.g., is_portrayed_on, may_indicate, adjacent_to, inside, surrounding, connected_by, supplied_by).\n"
        "  - `object`: The id of the second object involved in the relationship.\n\n"
    
        "3. **Including Relevant Medical Conditions or Diagnoses**:\n"
        "- Link any relevant medical conditions or diagnoses to the identified entities. Each condition should have:\n"
        "  - An `id` for identification.\n"
        "  - A `type` describing the condition (e.g., Certain types of disease).\n"
        "  - A `description` providing a brief explanation of the condition.\n\n"
    
        "4. **Highlighting Question Focus**:\n"
        "- Emphasize which parts of the scene graph are directly relevant to answering the question. This includes specific objects, relationships, or conditions that address the focus of the question.\n\n"
    
        "The JSON structure should follow this template:\n"
        "```json\n"
        "{\n"
        "  \"objects\": [...],\n"
        "  \"relationships\": [...],\n"
        "  \"conditions\": [...],\n"
        "  \"question_focus\": [\"focus_entity_or_relationship_ids\"]\n"
        "}\n"
        "```\n\n"
    
        "Ensure your output aligns closely with the specifics of the question and the medical domain knowledge required to interpret the image accurately."
    )
    
    return prompt


def get_scene_graph_refinement_prompt(initial_scene_graph, retrieved_knowledge):
    """Generate prompt for refining scene graph with knowledge base information"""
    prompt = (
        "You are a medical expert tasked with refining a medical scene graph using professional medical knowledge from established databases.\n\n"
        
        "**Initial Scene Graph:**\n"
        f"{initial_scene_graph}\n\n"
        
        "**Retrieved Medical Knowledge from RadGraph and TCGA-Reports:**\n"
        f"{retrieved_knowledge}\n\n"
        
        "Please carefully review the initial scene graph and refine it based on the retrieved professional medical knowledge. Your refinement should:\n\n"
        
        "1. **Verify Entity Definitions**: Check if the entities in the initial scene graph are correctly defined according to the professional medical knowledge.\n"
        
        "2. **Correct Entity Types**: Ensure entity types match standard medical terminology from the knowledge base.\n"
        
        "3. **Validate Relationships**: Verify that the relationships between entities are medically accurate based on the retrieved knowledge.\n"
        
        "4. **Add Missing Information**: Include any relevant entities or relationships that are supported by the knowledge base but missing from the initial scene graph.\n"
        
        "5. **Remove Incorrect Information**: Eliminate any entities or relationships that contradict the professional medical knowledge.\n"
        
        "6. **Enhance Attributes**: Update or add attributes based on standard medical descriptions from the knowledge base.\n\n"
        
        "Output the refined scene graph in the same JSON format as the initial scene graph, ensuring all modifications are grounded in the retrieved medical knowledge.\n"
        
        "Include a brief explanation of major changes made at the end of your response."
    )
    
    return prompt


def get_expert_consultation_prompt(question, description):
    """Generate prompt for expert consultation"""
    prompt_pt1 = (
        "You are a professional and experienced general practitioner. Please consider the following questions to be addressed and their corresponding medical scenarios and recommend several experts in different medical specialties to better answer the questions accurately.\n\n"
        "For each of the recommended experts, provide:\n"
        "- The expert's area of specialization.\n"
        "- List the specific expertise, skills, and knowledge that each expert brings to solving the problem. Emphasize how they will apply their specialized medical knowledge to address the problem thoroughly and accurately.\n"
        "When the expert is required, specify your request as: 'Expert: <specific task or information to extract>.'\n"
        "Repeat this format for each Expert.\n\n"
    )
    prompt_pt2 = (
        "When faced with a question about an image, the information in the question may not cover all the details of its description. Your task is to:\n\n"
        "- Provide a rationale for your approach to answering the question, explaining how you will use the images and the information provided by each expert to form a comprehensive answer.\n"
        "- Assign specific tasks to each expert, as needed, to gather additional information necessary to accurately answer the question, based on their abilities and the question to be answered. Ensure that each expert fully utilizes their expertise to provide the most professional reasoning and answers, focusing on their specific field.\n\n"
        "Your answer should be organized as follows:\n\n"
        "Answer: [Rationale: Explain your plan to interpret the question, including any initial insights based on the question and the image description provided. Describe how the expertise of each expert will be utilized to supplement this information and contribute to a comprehensive answer.]\n\n"
    )
    prompt_pt3 = (
        "experts_tasks:\n\n"
        "Expert: [Clearly list in detail the tasks that need to be completed by this expert.]\n"
        "If there are other Experts, the output format is similar.\n\n"
        "Make sure your answer follows this format to allow for consideration of different experts' analyses and to develop a systematic approach to solving the problem.\n"
        f"Please refer to the prompts and examples above to help me solve the following problem: {question}\n"
        f"Here is a medical scene graph of the related medical image: {description}"
    )
    return prompt_pt1 + prompt_pt2 + prompt_pt3


def get_expert_opinions_prompt(question, description, experts_tasks):
    """Generate prompt for expert opinions"""
    prompt = (
        "You are part of a team of medical experts who is good at answering medical questions,you have been assigned the specific task of solving the following problem.\n\n"
        f"Question: {question}\n"
        f"Medical Scene Graph: {description}\n"
        "Experts and their assigned tasks:\n"
        f"{experts_tasks}\n\n"
        "Please re-analyze the image carefully, give your answer and the reasons to support it, making full use of your expertise to provide the most professional and thorough analysis according to the task undertaken.\n"
        "Your answers should be organized as follows:\n"
        "Expert (Area of specialization):\n"
        "Reasoning and Answers: <Your job is to work with the other physicians on the team to provide answers to the questions. You will need to use your expertise to think through the problem step by step, give an answer to the question, and provide 2-3 sentences of reasoning to support the answer.>\n"
        "If there are other Experts, the output format is similar.\n\n"
        "Make sure your answer follows this format to allow for a comprehensive consideration of each expert's opinions, fully utilizing their specialized expertise to answer the question.\n"
    )
    return prompt


def get_expert_diagnosis_prompt(question, description, experts_opinions):
    """Generate prompt for expert diagnosis"""
    prompt = (
        "You are a professional and experienced medical diagnostic expert who is good at summarizing and synthesizing the opinions of multiple experts from different fields and giving an analysis report on the problem.\n\n"
        f"Below are some reports from experts in different medical fields.\n\n"
        f"{experts_opinions}\n\n"
        "You need to complete the following steps:\n"
        "1. Consider the following reports carefully and comprehensively.\n"
        "2. Extract key knowledge from the reports.\n"
        "3. Based on the knowledge, think again in combination with the problem and come up with a comprehensive and summary analysis.\n"
        "4. Your ultimate goal is to draw a refined and comprehensive report based on the following reports.\n\n"
        "You should output in exactly the same format:\n"
        "Key knowledge: <key knowledge>\n"
        "Overall analysis: <overall analysis>\n\n"
        f"Question: {question}\n"
        f"Medical Scene Graph: {description}\n"
        "Please provide your reasoning process, detailed reasons for your answer, and preliminary conclusion based on the information provided."
    )
    return prompt


def get_expert_evaluation_prompt(question, description, diagnosis_reasoning, experts_opinions):
    """Generate prompt for expert evaluation"""
    prompt = (
        "You are an experienced and professional medical reviewer who excels at reviewing the reasoning process of medical problems.\n"
        "Your task is to analyze and critique the reasoning process of the diagnostic specialist.\n"
        "If you find that the diagnostic expert's answer is inconsistent or in disagreement with a specialist's answer, you should:\n"
        "1. Talk to the specialist, exchange opinions, and ask them to rethink and provide their answer again with two sentences of supporting reasoning.\n"
        "2. After receiving the updated specialist opinions, talk to the diagnostic specialist again and ask them to rethink and update their reasoning and answers based on the updated opinions of the specialists.\n"
        "Your goal is to ensure that the diagnostic and specialty experts are in agreement and that their conclusions are based on sound medical knowledge and logical reasoning, agreed upon through iterative discussion.\n\n"
        f"Question: {question}\n"
        f"Medical Scene Graph: {description}\n"
        f"Diagnostic Specialist's Reasoning:\n{diagnosis_reasoning}\n"
        f"Specialist Experts' Opinions:\n{experts_opinions}\n\n"
        "Your response should be organized as follows:\n"
        "Review Analysis:\n"
        "<Your detailed analysis here>\n"
        "Feedback to Specialist Experts:\n"
        "<If applicable, provide feedback to specific Specialist Experts here, requesting them to rethink and provide updated reasoning in two sentences.>\n"
        "Feedback to Diagnostic Specialist:\n"
        "<After receiving updated specialists' opinions, provide feedback to the Diagnostic Specialist to rethink and update their reasoning and answers based on the new information.>\n"
        "If no feedback is necessary, state that all opinions are consistent.\n"
    )
    return prompt


def get_expert_evaluation_followup_prompt(question, description, updated_experts_opinions, diagnosis_reasoning):
    """Generate prompt for follow-up expert evaluation"""
    prompt = (
        "You are an experienced and professional medical reviewer who excels at reviewing the reasoning process of medical problems.\n"
        "Based on the updated opinions from the Specialist Experts, please analyze the Diagnostic Specialist's reasoning and provide feedback to ensure consistency and accuracy.\n"
        "If inconsistencies still exist, continue the iterative process of feedback and reassessment.\n"
        "Your goal is to ensure full agreement between the diagnostic and specialty experts, grounded in sound medical knowledge and logical reasoning.\n\n"
        f"Question: {question}\n"
        f"Medical Scene Graph: {description}\n"
        f"Diagnostic Specialist's Updated Reasoning:\n{diagnosis_reasoning}\n"
        f"Updated Specialist Experts' Opinions:\n{updated_experts_opinions}\n\n"
        "Your response should be organized as follows:\n"
        "Review Analysis:\n"
        "<Your detailed analysis here>\n"
        "Feedback to Diagnostic Specialist:\n"
        "<Provide feedback to the Diagnostic Specialist to rethink and update their reasoning and answers based on the updated specialists' opinions. Include two sentences of supporting reasoning.>\n"
        "If no further feedback is necessary, state that all opinions are consistent.\n"
    )
    return prompt


def get_specialists_rethink_prompt(question, description, feedback):
    """Generate prompt for specialists to rethink"""
    prompt = (
        "As a Specialist Expert, you have received feedback from the Review Expert regarding your previous analysis.\n"
        "Please carefully consider the feedback, re-evaluate your reasoning and conclusions, and provide an updated opinion.\n"
        "Your updated opinion should fully address the points raised in the feedback and utilize your specialized expertise to provide the most accurate and thorough analysis.\n\n"
        f"Question: {question}\n"
        f"Medical Scene Graph: {description}\n"
        f"Review Expert's Feedback:\n{feedback}\n\n"
        "Please provide your updated reasoning and answers, organized as follows:\n"
        "Updated Reasoning and Answers:\n"
        "<Your updated analysis here>\n"
    )
    return prompt


def get_diagnostic_reassessment_prompt(question, description, updated_experts_opinions):
    """Generate prompt for diagnostic reassessment"""
    prompt = (
        "As a professional and experienced medical diagnostic expert, you have received updated opinions from the Specialist Experts based on feedback from the Review Expert.\n"
        "Please re-evaluate the updated opinions, consider all the information carefully, and provide an updated analysis and conclusion.\n"
        "Your goal is to synthesize the experts' opinions and provide a comprehensive and accurate diagnosis.\n\n"
        f"Question: {question}\n"
        f"Medical Scene Graph: {description}\n"
        f"Updated Specialist Experts' Opinions:\n{updated_experts_opinions}\n\n"
        "Please provide your updated reasoning and conclusions, organized as follows:\n"
        "Updated Diagnostic Reasoning:\n"
        "<Your updated analysis here>\n"
    )
    return prompt


def get_integration_summary_prompt(question, description, expert_opinions):
    """Generate prompt for final integration summary"""
    prompt = (
        "You are a knowledgeable and skilled information integration medical expert. Please carefully consider all the experts' opinions provided, including those from the specialists, diagnostic expert, and review expert.\n"
        "Please gradually think and answer the questions based on the given question and experts' opinions information.\n"
        "Please note that we not only need answers, but more importantly, we need rationales for obtaining answers.\n"
        "Please prioritize using your knowledge to answer questions.\n"
        "Furthermore, please do not rely solely on supplementary information, as the provided supplementary information may not always be effective.\n"
        "Please do not answer with uncertainty; try your best to give an answer.\n"
        f"This is the question that needs to be answered: {question}\n"
        f"This is the refined medical scene graph of the related medical image: {description}\n"
        f"These are the opinions and reasoning of the experts:\n{expert_opinions}\n"
        f"The expected response format is as follows:\nInterpretation: <interpretation>\nAnswer: <answer>.\n"
    )
    return prompt