from evlearn.bundled.leanbase.base.funcs  import extract_name_kwargs

from .psee_evaluator import PseeEvaluator

EVALUATORS = {
    'psee'      : PseeEvaluator,
    'prophesee' : PseeEvaluator,
}

def select_evaluator(temporal):
    name, kwargs = extract_name_kwargs(temporal)

    if name not in EVALUATORS:
        raise ValueError(f"Unknown evaluator: '{name}'")

    return EVALUATORS[name](**kwargs)

def construct_evaluator(evaluator):
    evaluator = select_evaluator(evaluator)
    return evaluator

