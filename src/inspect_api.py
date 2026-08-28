from ragworkbench import DataLoaderFactory, DatasetName
import inspect

loader = DataLoaderFactory.create_loader(DatasetName.CLAP_NQ)
bm = loader.get_benchmark()
entries = bm.get_benchmark_entries()
e0 = entries[0]

# Inspect ground_truths_context_ids
print("=== ground_truths_context_ids type:", type(e0.ground_truths_context_ids))
if e0.ground_truths_context_ids:
    gid0 = e0.ground_truths_context_ids[0]
    print("=== GroundTruthContextId type:", type(gid0))
    print("=== GroundTruthContextId fields:", [m for m in dir(gid0) if not m.startswith("_")])
    if hasattr(gid0, "model_fields"):
        from ragworkbench.datasets_loader.data_models.rag_benchmark import GroundTruthContextId
        print("=== GroundTruthContextId model_fields:", GroundTruthContextId.model_fields)
    print("=== gid0 value:", gid0)

# Find Unitxt metric classes
try:
    from ragworkbench.evaluation import metrics as rwb_metrics
    print("=== ragworkbench.evaluation.metrics:", dir(rwb_metrics))
except Exception as e:
    print("No ragworkbench.evaluation.metrics:", e)

try:
    from ragworkbench.evaluation.evaluator import Evaluator
    print("=== Evaluator source snippet:")
    print(inspect.getsource(Evaluator)[:2000])
except Exception as e:
    print("Evaluator import error:", e)

try:
    import ragworkbench.evaluation
    print("=== ragworkbench.evaluation:", dir(ragworkbench.evaluation))
except Exception as e:
    print("ragworkbench.evaluation:", e)
