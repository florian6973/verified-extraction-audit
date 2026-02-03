import os
import random
import uuid
from omegaconf import DictConfig
import hydra
import warnings
from src.folder_handler import FolderHandler

@hydra.main(version_base=None, config_path="../../configs/jobs", config_name="submit_finetuning_job")
def main(cfg: DictConfig):
    # Convert config to args-like object for compatibility
    class Args:
        def __init__(self, config):
            self.base_model = config.base_model
            self.dataset_size = config.dataset_size
            self.n_epochs = config.n_epochs
            self.n_samples = config.n_samples
            self.lora = config.lora
            self.kg = config.kg
            self.pii_rate = config.pii_rate
            self.n_gpu = config.n_gpu
            self.n_cpu = config.n_cpu
            self.no_submit = config.no_submit
            self.base_path = config.base_path
            self.home_path = config.home_path
            self.backend = config.backend
            self.infer_only = config.infer_only
            self.injection_strategy = config.injection_strategy

    args = Args(cfg)
    cuda_visible_devices = ",".join([f"{i}" for i in range(args.n_gpu)])

    # for 1% dataset
    # 8B: 3h /epochs
    # 1B: 30 min /epochs
    time_min_per_epoch = 30
    if '8B' in args.base_model:
        time_min_per_epoch = 180
    duration = time_min_per_epoch*args.n_epochs*args.dataset_size + int(2*60*args.n_samples/1000)
    # duration = 120
    def minutes_to_slurm_time(minutes):
        days = minutes // (24 * 60)
        hours = (minutes % (24 * 60)) // 60
        mins = minutes % 60
        if days >= 30:
            days = 29
            warnings.warn(f"Days is greater than 30, setting to 29")
        return f"{days}-{hours:02d}:{mins:02d}:00"

    # careful jobname should depend on bash variables instead
    job_name = f"{args.base_model}-{args.dataset_size}-{args.pii_rate}-{args.kg}-{args.backend}"

    random_port = random.randint(10000, 65535)

    # Choose template based on backend
    if args.backend == "vllm":
        template_path = f"{args.base_path}/src/jobs/templates/finetuning_vllm-2.template"
    else:
        template_path = f"{args.base_path}/src/jobs/templates/finetuning_standard.template"

    # Read template file
    with open(template_path, 'r') as f:
        template_job = f.read()

    folder_handler = FolderHandler()
    dataset_id = folder_handler.query_dataset_unique(kwargs_filter={#"split": "train",
                                                                    "dataset_size": args.dataset_size,
                                                                    "pii_rate": args.pii_rate,
                                                                    "kg": "no-kg",
                                                                    "injection_strategy": args.injection_strategy,
                                                                    "name_strategy": "real",
                                                                    "sampling_strategy": "uniform"})
    print(dataset_id)
    random_identifier = str(uuid.uuid4())
    model_path = os.path.join(
        args.base_path,
        "outputs_models", "finetuning_auto",
        f"{args.base_model}-{args.dataset_size}-{args.pii_rate}-{args.kg}-{args.backend}-{args.injection_strategy}-{args.n_epochs}-{args.n_samples}-{random_identifier}",
    )
    new_model = {
        "model_name": args.base_model.split("-")[0],
        "model_size": args.base_model.split("-")[1],
        "dataset_id": dataset_id,
        "type": "instruct",
        # "status": "pending",
        "n_epochs": args.n_epochs,
        "model_path": model_path,
        "status": "training",
        "src_model_path": args.base_path + "/models/base/" + args.base_model,
    }
    model_id = folder_handler.add_model_to_index(new_model)
    # model_id = folder_handler.query_model_unique(kwargs_filter={"model_name": args.base_model,
    #                                                             "model_size": args.model_size,
    #                                                             "dataset_id": dataset_id,
    #                                                             "pii_rate": args.pii_rate,
    #                                                             "type": "instruct"})

    mem_per_gpu = 100
    to_replace = {
        "job_name": job_name,
        # "n_epochs": args.n_epochs,
        # "base_model": args.base_model,
        # "dataset_size": args.dataset_size,
        # "pii_rate": args.pii_rate,
        # "kg": "no-kg" if args.kg else "kg",
        "lora": args.lora,
        # "n_samples": args.n_samples,
        "n_gpu": args.n_gpu,
        "n_cpu": args.n_cpu,
        "mem": args.n_gpu * mem_per_gpu if args.dataset_size != 100 else 360,
        "duration": minutes_to_slurm_time(duration),
        "random_port": random_port,
        "base_path": args.base_path,
        "home_path": args.home_path,
        # "infer_only": args.infer_only,
        "cuda_visible_devices": cuda_visible_devices,
        "model_id": model_id,
    }
    job = template_job
    for key, value in to_replace.items():
        job = job.replace("{" + key + "}", str(value))
    print(job)

    if not args.no_submit:
        with open(f"{args.base_path}/src/jobs/finetuning/{job_name}.sh", "w") as f:
            f.write(job)

        # submit job
        os.system(f"sbatch {args.base_path}/src/jobs/finetuning/{job_name}.sh")

        print(f"Job {job_name} submitted")

        os.remove(f"{args.base_path}/src/jobs/finetuning/{job_name}.sh")

    else:
        job_short = job.split("\n")[12:]
        job_short = "#!/bin/bash\n" + "\n".join(job_short)
        print(job_short)

        with open(f"{args.base_path}/src/jobs/finetuning/tmp_{job_name}.sh", "w") as f:
            f.write(job_short)

if __name__ == "__main__":
    main()
