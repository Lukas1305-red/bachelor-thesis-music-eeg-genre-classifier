import json
import argparse


def train(mode="handcrafted", **embedding_options):
    """Run the original handcrafted pipeline or frozen foundation-embedding mode."""
    if mode == "foundation_embeddings":
        from code.classification.foundation_embedding_svm import (
            BioCodecEEGEmbeddingExtractor,
            evaluate_frozen_embeddings,
        )

        checkpoint = embedding_options.pop("checkpoint")
        model_name = embedding_options.pop("model_name")
        output = embedding_options.pop("output", "foundation_embedding_svm_results.json")
        extractor = BioCodecEEGEmbeddingExtractor(
            checkpoint_path=checkpoint,
            sample_rate=embedding_options.get("sample_rate", 250),
            name=model_name,
        )
        results = evaluate_frozen_embeddings(extractor=extractor, **embedding_options)
        with open(output, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"Saved foundation-embedding report to {output}!")
        return results
    if mode != "handcrafted":
        raise ValueError(f"Unknown headset classification mode: {mode}")

    # Keep the foundation-embedding mode usable without MLflow, which is only
    # needed by the legacy handcrafted classification module.
    from code.classification.classification import (
        FreqDomain,
        perform_subject_based_nested_cv_small_val,
    )

    print("Initialize")
    
    params = [
        {
            "selector__k": [5, 10, 20, 50], 
            "svm__kernel": ["linear"],
            "svm__C": [0.001, 0.01, 0.1, 1, 10, 100],
        },
        {
            "selector__k": [5, 10, 20, 50], 
            "svm__kernel": ["rbf", "sigmoid"],
            "svm__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "svm__gamma": ["scale", 0.001, 0.01, 0.1, 1], 
        }
    ]

    print("Training with Small-Holdout Nested CV (Option B)")
    
    results = perform_subject_based_nested_cv_small_val(
        param_grid=params,
        num_subjects=20,            
        deap=False,                 
        psd=FreqDomain.welch,       
        n_outer=5,                  
        n_inner_splits=8,      # Average over 8 random inner splits
        n_val_trials=3,        # Hold out exactly 3 trials for parameter validation
        random_state=1997
    )
    
    with open("headset_results_nested_cv.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved results to headset_results_nested_cv.json!")
    
    print("Finish")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["handcrafted", "foundation_embeddings"], default="handcrafted")
    parser.add_argument("--checkpoint", help="BioCodec EEG checkpoint; required for foundation_embeddings mode.")
    parser.add_argument("--model-name", help="BioCodec EEG architecture name; required for foundation_embeddings mode.")
    parser.add_argument("--output", default="foundation_embedding_svm_results.json")
    args = parser.parse_args()
    if args.mode == "foundation_embeddings":
        if not args.checkpoint or not args.model_name:
            parser.error("--checkpoint and --model-name are required for foundation_embeddings mode.")
        train(
            mode=args.mode,
            checkpoint=args.checkpoint,
            model_name=args.model_name,
            output=args.output,
        )
    else:
        train()
