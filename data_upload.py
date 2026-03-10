from huggingface_hub import login, upload_file

login()

upload_file(
    path_or_fileobj=r"G:\eunice-labs-exp\aria\aria_train.jsonl",
    path_in_repo="aria_train.jsonl",
    repo_id="Eunice-Labs/aria-math-reasoning-compressed",
    repo_type="dataset"
)