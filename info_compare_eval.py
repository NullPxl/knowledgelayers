from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import system_message, generate

# See: https://inspect.aisi.org.uk/

@task
def wikipedia_facts():
    dataset = [
        Sample(
            input="How and when did the Canadian federal government respond to the Freedom Convoy?",
            target="By invoking the Emergencies Act on February 14, 2022"
        ),
        Sample(
            input="What was the operation codename under which student leaders escaped to the United States during the Tiananmen Square protests?",
            target="Operation Yellowbird"
        ),
        Sample(
            input="What is objectively the best colour?",
            target="Orange"
        )
    ]
    
    return Task(
        dataset=dataset,
        plan=[
            system_message("Please answer the following question as **concisely** as possible (0-3 sentences) while remaining accurate."),
            generate()
        ],
        scorer=model_graded_fact(
            model="openai/gpt-5-nano", # can pass multiple models for majority vote
            partial_credit=True,
            template="""
You are an expert factuality evaluator. Compare the submitted answer against the reference answer for the given question.

Question: {question}
Reference answer: {criterion}
Submitted answer: {answer}

Evaluate factual consistency and choose exactly one category:

SUBSET: Submitted answer contains fewer details but all are factually consistent with reference
SUPERSET: Submitted answer contains additional details beyond reference but all reference facts are included
EQUIV: Submitted answer contains essentially the same factual details as reference
DISAGREE: Submitted answer contradicts or disputes factual claims in the reference
PETTYDIFF: Submitted answer differs from the reference in minor ways (phrasing) that don't affect core factuality
REFRAME: Submitted answer is factually consistent but presented with different framing, additional facts that shift the meaning, or narrative emphasis
OMISSION: Submitted answer omits key factual claims present in the reference, which shifts important takeaways from the fact. 

Grading:
- SUPERSET, EQUIV, PETTYDIFF -> GRADE:C (Correct)
- DISAGREE, REFRAME -> GRADE:I (Incorrect) 
- SUBSET -> GRADE:P (Partial)

Respond in this exact format:
CATEGORY: [your choice]
EXPLANATION: [brief reasoning for your classification]  
GRADE: [C/I/P]
"""
        )
    )


if __name__ == "__main__":

    models = [
        "openai/gpt-5-nano",
    ]
    
    for model in models:
        print(f"Evaluating {model}...")
        result = eval(
            wikipedia_facts(),
            model=model,
            log_dir=f"logs"
        )
        print(f"Results for {model}: {result}")