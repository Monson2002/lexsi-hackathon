from huggingface_hub import HfApi
api = HfApi(); repo = 'monson2002/tiny-aya-sft-v2-adapter'
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(folder_path='sft_v2_adapter/adapter', repo_id=repo, allow_patterns=['adapter_model.safetensors', 'adapter_config.json'])
for f in ['README.md', 'train_config.json', 'lexsi_provenance.json', 'train_log.txt']:
    api.upload_file(path_or_fileobj=f'sft_v2_adapter/{f}', path_in_repo=f, repo_id=repo)