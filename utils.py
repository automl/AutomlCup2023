from datetime import datetime


def extend_results(base_results: dict, new_results: dict, check_id: bool = False) -> dict:
    """
    Adds new result to base result.
    """
    for suite, datasets in new_results.items():
        for dataset, run_results in datasets.items():
            base_dataset_results = base_results.get(suite, {}).get(dataset, None)
            if base_dataset_results is None:
                # dataset is not in base results
                new_dataset_results = {suite: {dataset: run_results}}
                base_results = deep_update(base_results, new_dataset_results)
            else:
                # dataset is in base results
                if check_id:
                    # remove base duplicate results based on id
                    new_ids = [run_result['id'] for run_result in run_results]
                    base_dataset_results = [val for val in base_dataset_results if val['id'] not in new_ids]
                    base_results[suite][dataset] = base_dataset_results

                base_dataset_results.extend(run_results)
            # sort by date
            # base_results[suite][dataset].sort(key= lambda x: datetime.strptime(x['date'], "%Y-%m-%d %H:%M:%S"))
    return base_results


def deep_update(mapping, *updating_mappings):
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if k in updated_mapping and isinstance(updated_mapping[k], dict) and isinstance(v, dict):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping
