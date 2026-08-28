from ragworkbench import DataLoaderFactory
from ragbenchlab.configs.config import get_scenario
from ragworkbench.datasets_loader.data_models.data_sampling_params import DataSamplingParams

def main():
    scenario = get_scenario('nq')
    loader = DataLoaderFactory.create_loader(
        scenario.dataset_name, 
        sampling_params=DataSamplingParams(question_limit=1), 
        cache_dir='cache'
    )
    
    try:
        benchmark = loader.get_benchmark()
        entry = benchmark.benchmark_entries[0]
        print("entry type:", type(entry))
        
        # print specific fields if they exist
        fields_to_check = ['question_id', 'question', 'ground_truth_answers', 'ground_truths_context_ids', 'ground_truth_context_ids']
        
        for field in fields_to_check:
            val = getattr(entry, field, "NOT_FOUND")
            print(f"{field}: {val}")
            
    except Exception as e:
        print("Failed:", e)

if __name__ == '__main__':
    main()
